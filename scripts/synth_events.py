"""
Synthetic hydrograph & hyetograph generator (random-scaled resampling).

This script creates *num_synthetic* synthetic hydrographs and hyetographs by:
1) Randomly selecting one **base** column (series) from each input table.
2) Scaling the entire selected series by a random factor in a given range
   (default: [0.8, 1.2]).
3) Repeating until the requested number of synthetic series are produced.

Outputs
-------
- Tab-separated files with columns T1..T<num_synthetic>.
- Column count equals the number of synthetic series.
- Row count equals the number of timesteps in the **base** inputs.

Notes
-----
- This is a simple data augmentation baseline (global gain change).
  You may extend it with shifts, temporal warping, noise, etc.
- The generator treats each hydrograph / hyetograph independently.
- No assumptions are made about time spacing; values are scaled as-is.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def _read_table(path: str | Path, sep: str = "\t") -> pd.DataFrame:
    """Read a whitespace- or delimiter-separated table into a DataFrame.

    Parameters
    ----------
    path : str | Path
        File path to read.
    sep : str, default "\\t"
        Field delimiter. Use "\\t" for TSV; for generic whitespace use regex: r"\\s+".

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (timesteps, n_series).

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file is empty or has no columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, sep=sep, engine="python")
    if df.shape[1] == 0:
        raise ValueError(f"No columns detected in: {path}")
    return df


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as a tab-separated file with headers (no index)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, header=True)


# ------------------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------------------

def generate_synthetic_data(
    base_hydrographs: pd.DataFrame,
    base_hyetographs: pd.DataFrame,
    num_synthetic: int,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic hydrographs and hyetographs by global scaling.

    For each synthetic series:
      - choose one random **column** from the base table
      - multiply the entire column by a random factor in `scale_range`

    Parameters
    ----------
    base_hydrographs : pd.DataFrame
        Base hydrograph table of shape (T, K_h), where T is timesteps.
    base_hyetographs : pd.DataFrame
        Base hyetograph table of shape (T, K_p).
    num_synthetic : int
        Number of synthetic series to produce for each table.
    scale_range : (float, float), default (0.8, 1.2)
        Inclusive lower/upper bounds for the random multiplicative factor.
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility. If None, a default RNG is used.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Two DataFrames with shape (T, num_synthetic). Columns are named T1..Tn.

    Raises
    ------
    ValueError
        If input tables are empty or `num_synthetic` < 1.
    """
    if num_synthetic < 1:
        raise ValueError("num_synthetic must be >= 1")
    if base_hydrographs.shape[1] == 0 or base_hyetographs.shape[1] == 0:
        raise ValueError("Base tables must contain at least one column.")

    rng = rng or np.random.default_rng()

    n_h_cols = base_hydrographs.shape[1]
    n_p_cols = base_hyetographs.shape[1]

    syn_hydro = []
    syn_hyeto = []

    for _ in range(num_synthetic):
        # ---- Hydrograph ----
        col_h = int(rng.integers(0, n_h_cols))
        factor_h = float(rng.uniform(scale_range[0], scale_range[1]))
        series_h = base_hydrographs.iloc[:, col_h].to_numpy(dtype=float) * factor_h
        syn_hydro.append(series_h)

        # ---- Hyetograph ----
        col_p = int(rng.integers(0, n_p_cols))
        factor_p = float(rng.uniform(scale_range[0], scale_range[1]))
        series_p = base_hyetographs.iloc[:, col_p].to_numpy(dtype=float) * factor_p
        syn_hyeto.append(series_p)

    # Stack lists of arrays into (num_synthetic, T) then transpose to (T, num_synthetic)
    hydro_df = pd.DataFrame(np.vstack(syn_hydro).T)
    hyeto_df = pd.DataFrame(np.vstack(syn_hyeto).T)

    # Human-friendly column names: T1..T<num_synthetic>
    new_cols = [f"T{i}" for i in range(1, num_synthetic + 1)]
    hydro_df.columns = new_cols
    hyeto_df.columns = new_cols

    return hydro_df, hyeto_df


# ------------------------------------------------------------------------------
# Script usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # --------------------------- User parameters ---------------------------- #
    # Base input files (TSV with columns = distinct series)
    hydrograph_file_path = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Base_Hydrographs.txt")
    hyetograph_file_path = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Base_Hyetographs.txt")

    # Number of synthetic series to generate for each table
    num_synthetic = 25

    # Random scaling range (inclusive bounds)
    scale_range = (0.8, 1.2)

    # Optional reproducibility
    rng = np.random.default_rng(seed=1234)

    # Output locations
    hydrograph_save_dir = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Synthetic_Hydrographs")
    hyetograph_save_dir = Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Case_3\Synthetic_Hyetographs")
    synthetic_hydrograph_file_path = hydrograph_save_dir / "Hydrographs_Test.txt"
    synthetic_hyetograph_file_path = hyetograph_save_dir / "Hyetographs_Test.txt"

    # ------------------------------- Pipeline ------------------------------- #
    base_hydrographs = _read_table(hydrograph_file_path, sep="\t")
    base_hyetographs = _read_table(hyetograph_file_path, sep="\t")

    syn_hydro_df, syn_hyeto_df = generate_synthetic_data(
        base_hydrographs=base_hydrographs,
        base_hyetographs=base_hyetographs,
        num_synthetic=num_synthetic,
        scale_range=scale_range,
        rng=rng,
    )

    _write_table(syn_hydro_df, synthetic_hydrograph_file_path)
    _write_table(syn_hyeto_df, synthetic_hyetograph_file_path)

    print(f"Synthetic hydrographs saved to: {synthetic_hydrograph_file_path}")
    print(f"Synthetic hyetographs saved to: {synthetic_hyetograph_file_path}")
