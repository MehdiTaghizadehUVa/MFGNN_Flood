"""
HEC-RAS batch runner & extractor for generating training/test datasets.

This script automates three tasks around a 2D HEC-RAS project:

1) **Hydrograph injection**: Overwrites the `Muncie2D_SI.u01` plan file with a
   pair of hydrographs (upstream flow + precipitation) pulled from tabular
   text files. Values are formatted as fixed-width, right-aligned columns that
   match HEC-RAS expectations.

2) **Simulation control**: Launches HEC-RAS via the COM automation interface,
   opens the specified project, computes the current plan, and blocks until
   the computation is finished. Returns wall-clock runtime.

3) **Result extraction**: Reads the output `.hdf` file and exports selected
   arrays (geometry, static rasters, and unsteady time series) as tab-
   separated `.txt` files for downstream ML workflows.

Outputs (per run)
-----------------
- `XY`, `CE`, `CA`, `N` (written once on the first run)
- `WD_<id>`, `VX_<id>`, `VY_<id>`, `US_InF_<id>`, `DS_OuF_<id>`
- `computation_times_<prefix>.txt` with timing logs

Assumptions
-----------
- Windows host with HEC-RAS 6.5 installed (ProgID: `"RAS65.HECRASController"`).
  Adjust the ProgID if you run a different version (e.g., `"RAS64.HECRASController"`).
- The .u01 file contains single sections labeled literally:
  - `Flow Hydrograph=`
  - `Precipitation Hydrograph=`
  The next lines are the numeric rows to be replaced.

Safety notes
------------
- The script **overwrites** portions of the `.u01` file in-place. Keep a copy.
- Paths are Windows examples; update to your environment.

"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import numpy as np
import pandas as pd

try:
    import win32com.client  # pywin32
except Exception as e:  # pragma: no cover
    win32com = None  # type: ignore
    _WIN32COM_IMPORT_ERROR = e


# ------------------------------------------------------------------------------
# Hydrograph I/O & formatting
# ------------------------------------------------------------------------------

def read_hydrographs(file_path: str | Path) -> pd.DataFrame:
    """Load a whitespace-delimited hydrograph table.

    The table is assumed to have one column per hydrograph and one row per time
    step. No header interpretation is enforced beyond pandas defaults.

    Parameters
    ----------
    file_path : str | Path
        Path to a text file with whitespace-separated numeric values.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, K) where T is timesteps and K hydrographs.

    Notes
    -----
    Using a DataFrame lets you select a column (by name) and then format it
    with `format_hydrograph_values`.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Hydrograph file not found: {file_path}")
    # sep with regex \s+ avoids deprecation warning for delim_whitespace
    return pd.read_csv(file_path, sep=r"\s+", engine="python")


def format_hydrograph_values(values: Sequence[float | int | str]) -> List[str]:
    """Format a sequence of values into **8-char, right-aligned** strings.

    HEC-RAS plan files expect numeric hydrograph values in fixed-width fields.
    This helper:
      - casts to float
      - formats with up to 6 significant digits
      - truncates strings longer than 8 characters
      - right-aligns to width 8

    Parameters
    ----------
    values : Sequence[float | int | str]
        Values in time order.

    Returns
    -------
    List[str]
        A list of strings, each exactly width 8, ready to be concatenated
        into rows (e.g., 10 values per line → 80-char lines).
    """
    out: List[str] = []
    for v in values:
        # Robust cast to float then compact string representation
        s = f"{float(v):.6g}"
        # Guard fixed-width (8 chars max)
        if len(s) > 8:
            s = s[:8]
        out.append(f"{s:>8}")
    return out


# ------------------------------------------------------------------------------
# .u01 manipulation
# ------------------------------------------------------------------------------

def _replace_block(lines: List[str], marker: str, new_rows: List[str]) -> bool:
    """Replace the block of numeric rows following a marker line.

    The function finds the **first** line that contains `marker`, then replaces
    the subsequent `len(new_rows)` lines with the provided list.

    Parameters
    ----------
    lines : list[str]
        Entire file loaded as list of lines.
    marker : str
        Substring to search for (e.g., ``"Flow Hydrograph="``).
    new_rows : list[str]
        New lines to insert (each should end with ``"\\n"``).

    Returns
    -------
    bool
        True if a replacement was performed, False if the marker was not found.
    """
    for i, line in enumerate(lines):
        if marker in line:
            start = i + 1
            end = start + len(new_rows)
            lines[start:end] = new_rows
            return True
    return False


def modify_u01_file(u01_file_path: str | Path, new_flow_data: List[str], new_precipitation_data: List[str]) -> None:
    """Overwrite the flow and precipitation hydrograph blocks in a `.u01` file.

    Parameters
    ----------
    u01_file_path : str | Path
        Path to the plan input file (e.g., `Muncie2D_SI.u01`).
    new_flow_data : list[str]
        Prepared lines for the **Flow Hydrograph** block (end with `\\n`).
    new_precipitation_data : list[str]
        Prepared lines for the **Precipitation Hydrograph** block (end with `\\n`).

    Raises
    ------
    RuntimeError
        If expected markers are not found in the file.
    """
    u01_file_path = Path(u01_file_path)
    lines = u01_file_path.read_text().splitlines(keepends=True)

    ok_flow = _replace_block(lines, "Flow Hydrograph=", new_flow_data)
    ok_prec = _replace_block(lines, "Precipitation Hydrograph=", new_precipitation_data)

    if not ok_flow or not ok_prec:
        missing = []
        if not ok_flow:
            missing.append("Flow Hydrograph=")
        if not ok_prec:
            missing.append("Precipitation Hydrograph=")
        raise RuntimeError(f"Missing required marker(s) in {u01_file_path.name}: {', '.join(missing)}")

    u01_file_path.write_text("".join(lines))


# ------------------------------------------------------------------------------
# HEC-RAS compute
# ------------------------------------------------------------------------------

def run_hec_ras(project_dir: str | Path, project_name: str = "Muncie2D_SI.prj", prog_id: str = "RAS65.HECRASController") -> float:
    """Compute the *current plan* for a HEC-RAS project via COM automation.

    Parameters
    ----------
    project_dir : str | Path
        Folder containing the HEC-RAS project.
    project_name : str, default "Muncie2D_SI.prj"
        Project file name.
    prog_id : str, default "RAS65.HECRASController"
        COM ProgID for the installed HEC-RAS version. Adjust for your install
        (e.g., "RAS64.HECRASController").

    Returns
    -------
    float
        Wall-clock computation time in seconds.

    Raises
    ------
    RuntimeError
        If COM automation is unavailable or computation fails.
    """
    if win32com is None:  # pragma: no cover
        raise RuntimeError(f"pywin32 not available: {_WIN32COM_IMPORT_ERROR!r}")

    project_dir = Path(project_dir)
    prj_path = project_dir / project_name
    if not prj_path.exists():
        raise FileNotFoundError(f"HEC-RAS project not found: {prj_path}")

    # Launch controller (Consider DispatchEx if you need isolation per run)
    ras = win32com.client.Dispatch(prog_id)

    try:
        ras.Project_Open(prj_path.as_posix())
        t0 = time.time()

        # Start compute for the *current* plan configured in the GUI/project.
        # Arguments: (bShowComputationWindow, bBlocked, bShowController)
        ras.Compute_CurrentPlan(None, None, False)

        # Poll for completion; Compute_Complete returns 1 when finished
        while True:
            status = ras.Compute_Complete()
            if status == 1:
                break
            time.sleep(5.0)

        dt = time.time() - t0
    finally:
        # Ensure the controller is closed even on exceptions
        try:
            ras.Project_Close()
        finally:
            ras.QuitRAS()

    return dt


# ------------------------------------------------------------------------------
# Result extraction (.hdf → .txt)
# ------------------------------------------------------------------------------

def extract_and_save_data(hdf5_file: str | Path, identifier: str, first_run: bool, save_dir: str | Path, prefix: str) -> None:
    """Export geometry, static rasters, and unsteady series from a HEC-RAS HDF5.

    The function writes the following files to `save_dir`:

    Static (written only on the **first** call when `first_run=True`)
      - `{prefix}_XY.txt` : cell center coordinates
      - `{prefix}_CE.txt` : min cell elevation
      - `{prefix}_CA.txt` : cell surface area
      - `{prefix}_N.txt`  : Manning's n at cell centers

    Time series (written every call)
      - `{prefix}_WD_{identifier}.txt`     : cell invert depth (water depth)
      - `{prefix}_VX_{identifier}.txt`     : cell velocity X
      - `{prefix}_VY_{identifier}.txt`     : cell velocity Y
      - `{prefix}_US_InF_{identifier}.txt` : upstream boundary inflow
      - `{prefix}_DS_OuF_{identifier}.txt` : downstream boundary outflow

    Parameters
    ----------
    hdf5_file : str | Path
        Path to HEC-RAS output `.hdf` produced by the compute.
    identifier : str
        Tag appended to time-series outputs; typically the hydrograph column name.
    first_run : bool
        If True, write static geometry once; subsequent calls can set False to skip.
    save_dir : str | Path
        Output directory for all exported `.txt` files.
    prefix : str
        A short tag indicating resolution (e.g., `"M10"` or `"M80"`).

    Notes
    -----
    HDF5 dataset paths reflect HEC-RAS 2D structures. Adjust if your project
    uses different names (e.g., multiple 2D areas or different perimeter names).
    """
    hdf5_file = Path(hdf5_file)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_file, "r") as f:
        # Geometry / static
        xy = f["Geometry/2D Flow Areas/Cell Points"][:]  # (N, 2)
        elev = f["Geometry/2D Flow Areas/Perimeter 1/Cells Minimum Elevation"][:]
        area = f["Geometry/2D Flow Areas/Perimeter 1/Cells Surface Area"][:]
        mann = f["Geometry/2D Flow Areas/Perimeter 1/Cells Center Manning's n"][:]

        # Unsteady time series
        wd = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Invert Depth"][:]  # (T, N)
        vx = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity X"][:]  # (T, N)
        vy = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y"][:]  # (T, N)

        # Boundary conditions (timeseries)
        inflow = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Boundary Conditions/BC_UPS"][:]      # (T, 2) often [time, value]
        outflow = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Boundary Conditions/BC_DownS"][:]   # (T, 2)

    # Save with high precision; downstream can reduce or renormalize as needed
    fmt = "%.9f"

    # Static exports once (XY/CE/CA/N)
    if first_run:
        np.savetxt(save_dir / f"{prefix}_XY.txt", xy, delimiter="\t", fmt=fmt)
        np.savetxt(save_dir / f"{prefix}_CE.txt", elev, delimiter="\t", fmt=fmt)
        np.savetxt(save_dir / f"{prefix}_CA.txt", area, delimiter="\t", fmt=fmt)
        np.savetxt(save_dir / f"{prefix}_N.txt", mann, delimiter="\t", fmt=fmt)

    # Time series per identifier
    np.savetxt(save_dir / f"{prefix}_WD_{identifier}.txt", wd, delimiter="\t", fmt=fmt)
    np.savetxt(save_dir / f"{prefix}_VX_{identifier}.txt", vx, delimiter="\t", fmt=fmt)
    np.savetxt(save_dir / f"{prefix}_VY_{identifier}.txt", vy, delimiter="\t", fmt=fmt)
    np.savetxt(save_dir / f"{prefix}_US_InF_{identifier}.txt", inflow, delimiter="\t", fmt=fmt)
    np.savetxt(save_dir / f"{prefix}_DS_OuF_{identifier}.txt", outflow, delimiter="\t", fmt=fmt)


# ------------------------------------------------------------------------------
# CLI / Batch loop
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------- USER INPUTS ------------------------------ #
    prefix = "M10"  # model resolution tag for outputs

    project_path = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Base_Model_M10")
    u01_file_path = project_path / "Muncie2D_SI.u01"

    flow_hydrograph_file = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Synthetic_Hydrographs\Hydrographs_Train.txt")
    precipitation_hydrograph_file = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Synthetic_Hyetographs\Hyetographs_Train.txt")

    save_dir = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Results_M80")
    base_hdf5_path = project_path / "Muncie2D_SI.p02.hdf"

    computation_times_file = save_dir / f"computation_times_{prefix}.txt"

    # Range of columns (inclusive) from the hydrograph tables to iterate over
    start_column = 1  # 0-based index (excluding time col if present)
    end_column = 5

    # ------------------------------ PREP WORK ------------------------------- #
    save_dir.mkdir(parents=True, exist_ok=True)

    flow_hydrographs_df = read_hydrographs(flow_hydrograph_file)
    precipitation_hydrographs_df = read_hydrographs(precipitation_hydrograph_file)

    first_run = True  # write static geometry only for the first simulation

    # ------------------------------ MAIN LOOP ------------------------------- #
    for idx in range(start_column, end_column + 1):
        # Column selection by integer position (consistent with the user's code)
        col_name = flow_hydrographs_df.columns[idx]

        # Prepare 8-char, right-aligned value strings and pack into 80-char lines
        # HEC-RAS typically expects 10 values per row for these blocks
        flow_vals = format_hydrograph_values(flow_hydrographs_df[col_name].tolist())
        flow_lines = ["".join(flow_vals[i : i + 10]) + "\n" for i in range(0, len(flow_vals), 10)]

        precip_vals = format_hydrograph_values(precipitation_hydrographs_df[col_name].tolist())
        precip_lines = ["".join(precip_vals[i : i + 10]) + "\n" for i in range(0, len(precip_vals), 10)]

        # Patch the .u01 plan file with the new hydrographs
        modify_u01_file(u01_file_path, flow_lines, precip_lines)

        # Run the simulation via COM automation and record elapsed time
        dt = run_hec_ras(project_path)

        with computation_times_file.open("a") as f:
            f.write(f"Hydrograph {col_name}: {dt:.2f} seconds\n")

        # Export relevant arrays from the HDF5 results
        extract_and_save_data(base_hdf5_path, identifier=str(col_name), first_run=first_run, save_dir=save_dir, prefix=prefix)
        first_run = False  # static exports done

        print(f"[OK] Completed simulation and extraction for hydrograph '{col_name}'")

    print("All simulations and data extractions complete.")
