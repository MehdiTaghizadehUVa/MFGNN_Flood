# inference.py
# Evaluation, metrics, and plots after training.

"""
Inference & evaluation utilities for two-stage (LF→HF) MeshGraphNet models.

This module performs end-to-end **post-training evaluation**:

1) Loads the best LF and HF checkpoints.
2) Builds normalized LF/HF test graphs (using saved stats).
3) Augments HF graphs with an **interpolated LF prediction** as the last node
   feature (see `update_data` in `utils.py`).
4) Runs HF inference and computes regression & classification-style metrics:
   - MAE / MSE / RMSE / RRMSE
   - CSI at 0.05 m and 0.30 m thresholds
   - Continuous CSI (tolerance-based score)
   - Average per-sample inference time
5) Saves metrics to CSV and produces diagnostic plots:
   - Ground truth, prediction, upsampled LF, absolute error map
   - Histogram of average node-wise errors over the test set
6) Saves raw per-node error arrays for future offline analysis.

Artifacts
---------
- `<ckpt_path>/<ckpt_name_hf>/test_metrics.csv`
- `<ckpt_path>/<results_dir>/*.png` (visualizations)
- `<ckpt_path>/<results_dir>/*errors.npy` (error arrays)

Requirements
------------
- A `Constants` configuration object (see `constants.py`).
- Trained best checkpoints saved by the training script:
  - `<ckpt_path>/<ckpt_name_lf>/best_model_LF/MeshGraphNet.0.0.mdlus`
  - `<ckpt_path>/<ckpt_name_hf>/best_model_HF/MeshGraphNet.0.0.mdlus`
- Normalization JSONs:
  - `normalization_params_LF.json` and `normalization_params_HF.json`
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection

from constants import Constants
from utils import load_model, process_test_folder, update_data


def plot_mesh(
    ax,
    g: dgl.DGLGraph,
    values: np.ndarray | torch.Tensor,
    cmap: str,
    title: str,
    vmin=None,
    vmax=None,
) -> None:
    """Render a mesh-style node scatter overlaid with edge segments.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    g : dgl.DGLGraph
        Graph whose node features' first two columns contain (x, y) coordinates.
        Edges are drawn as thin segments for visual context.
    values : np.ndarray | torch.Tensor
        Per-node scalar to color nodes by (same order as graph nodes).
    cmap : str
        Matplotlib colormap name (e.g., "viridis").
    title : str
        Title for the subplot (unused in minimalist layout; kept for clarity).
    vmin, vmax : float, optional
        Color limits. If not provided, Matplotlib will infer limits.

    Notes
    -----
    - This function does **not** draw a color bar to keep the figure uncluttered.
    - Axis labels and frames are removed for a clean, map-like appearance.
    """
    # Extract node coordinates (assumed to be stored in the first two feature columns)
    coords = g.ndata["x"][:, :2].cpu().numpy()

    # Build a list of line segments from edge indices for a light mesh context
    e_src, e_dst = g.edges()
    e_idx = torch.stack((e_src, e_dst)).cpu().numpy()
    segs = np.stack((coords[e_idx[0]], coords[e_idx[1]]), axis=1)
    ax.add_collection(LineCollection(segs, colors="k", linewidths=0.01))

    # Convert tensor inputs to numpy for Matplotlib
    vals = values.detach().cpu().numpy() if torch.is_tensor(values) else values

    # Node scatter; thin black edges around points help separate dense regions
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=vals,
        cmap=cmap,
        s=30,
        edgecolor="k",
        lw=0.01,
        vmin=vmin,
        vmax=vmax,
    )

    # Tighten view to data extent and hide axes for a clean map
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_aspect(0.8)
    ax.axis("off")


def compute_csi(pred: torch.Tensor, tgt: torch.Tensor, threshold: float = 0.05) -> float:
    """Compute the (binary) Critical Success Index (CSI) at a depth threshold.

    CSI = hits / (hits + misses + false_alarms)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted water depths (denormalized, meters).
    tgt : torch.Tensor
        Ground-truth water depths (denormalized, meters).
    threshold : float, default 0.05
        Flood depth threshold (meters). Nodes above this are considered "wet".

    Returns
    -------
    float
        CSI score in [0, 1]. Higher is better.
    """
    hits = ((pred >= threshold) & (tgt >= threshold)).float().sum().item()
    miss = ((pred < threshold) & (tgt >= threshold)).float().sum().item()
    fa = ((pred >= threshold) & (tgt < threshold)).float().sum().item()
    den = hits + miss + fa
    return hits / den if den > 0 else 0.0


def compute_continuous_csi(pred: torch.Tensor, tgt: torch.Tensor, delta: float = 0.05) -> float:
    """Continuous variant of CSI using a tolerance band around ground truth.

    For each node:
      score_i = max(0, 1 - |pred_i - tgt_i| / delta)

    The final score is the mean over nodes. This rewards near-misses rather than
    strictly thresholding. Set `delta` to the error tolerance (meters).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted depths (denormalized).
    tgt : torch.Tensor
        Ground-truth depths (denormalized).
    delta : float, default 0.05
        Tolerance (meters) for full credit.

    Returns
    -------
    float
        Mean continuous CSI in [0, 1]. Higher is better.
    """
    eps = 1e-8
    err = torch.abs(pred - tgt)
    score = torch.clamp(1.0 - err / max(delta, eps), min=0.0)
    return torch.mean(score).item()


def evaluate_and_plot() -> None:
    """Evaluate the trained HF model on a random test subset and generate plots.

    Pipeline
    --------
    1) Load best LF and HF models from checkpoints.
    2) Sample `C.N_Test` HF scenarios from `C.test_dir`.
    3) Build normalized LF/HF test graphs with saved stats.
    4) Interpolate LF predictions onto HF nodes (append as last feature).
    5) Run HF inference; denormalize predictions/targets using HF stats.
    6) Compute metrics per sample and write to CSV.
    7) Summarize metrics (mean ± std) on stdout.
    8) Save per-sample diagnostic plots and node-wise error distributions.

    Notes
    -----
    - The random test subset is reproducible via NumPy's `default_rng(42)`.
    - Assumes the first two node features are (x, y) coordinates.
    """
    C = Constants()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- Load models ------------------------------ #
    lf_best = Path(C.ckpt_path) / C.ckpt_name_lf / "best_model_LF" / "MeshGraphNet.0.0.mdlus"
    hf_best = Path(C.ckpt_path) / C.ckpt_name_hf / "best_model_HF" / "MeshGraphNet.0.0.mdlus"
    low_model = load_model(lf_best, device, "MeshGraphNet_LF")
    high_model = load_model(hf_best, device, "MeshGraphNet_HF")

    # ------------------------------ Test set -------------------------------- #
    # Collect all HF scenario IDs from files named like: M10_WD_<ID>.txt
    all_ids = sorted([Path(f).stem.split("_")[2] for f in Path(C.test_dir).iterdir() if f.name.startswith("M10_WD_")])

    # Reproducible random subset of size C.N_Test
    rng = np.random.default_rng(42)
    ids_test = sorted(rng.choice(all_ids, size=C.N_Test, replace=False))

    # Build normalized test graphs for both fidelities
    test_hf = process_test_folder(C.test_dir, ids_test, is_low_fidelity=False, cache_dir=C.cache_dir)
    test_lf = process_test_folder(C.test_dir, ids_test, is_low_fidelity=True, cache_dir=C.cache_dir)

    # Convert mapping → list preserving order of ids_test
    data_hf = list(test_hf.values())
    data_lf = list(test_lf.values())

    # Append interpolated LF scalar as the last HF feature channel
    proc = update_data(data_hf, data_lf, low_model, device)

    # -------------------------- Denormalization ----------------------------- #
    # Load HF label stats to restore physical units (meters)
    stats_hf = json.loads((Path(C.ckpt_path) / C.ckpt_name_hf / "normalization_params_HF.json").read_text())
    lo = torch.tensor(stats_hf["label_min"], device=device)
    hi = torch.tensor(stats_hf["label_max"], device=device)

    # ------------------------------ Metrics --------------------------------- #
    out_csv = Path(C.ckpt_path) / C.ckpt_name_hf / "test_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mae_l: List[float] = []
    mse_l: List[float] = []
    rmse_l: List[float] = []
    rrmse_l: List[float] = []
    c05_l: List[float] = []
    c03_l: List[float] = []
    ccont_l: List[float] = []
    tt_l: List[float] = []

    # Evaluate each test graph and log per-sample metrics to CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Sample Index",
                "MAE",
                "MSE",
                "RMSE",
                "RRMSE",
                "CSI_0.05",
                "CSI_0.3",
                "Continuous_CSI",
                "Test Time",
            ],
        )
        writer.writeheader()

        for i, g in enumerate(proc):
            g = g.to(device)

            # Forward pass timing (pure inference latency per sample)
            t0 = time.time()
            with torch.no_grad():
                p = high_model(g.ndata["x"], g.edata["x"], g)
            tt = time.time() - t0

            # Denormalize prediction and target to meters
            y = g.ndata["y"]
            p = p * (hi - lo) + lo
            y = y * (hi - lo) + lo

            # Regression metrics
            mae = F.l1_loss(p, y).item()
            mse = F.mse_loss(p, y).item()
            rmse = torch.sqrt(F.mse_loss(p, y)).item()
            rrmse = (torch.linalg.vector_norm(p - y) / (torch.linalg.vector_norm(y) + 1e-8)).item()

            # Classification-style scores on water depth exceedance
            c05 = compute_csi(p, y, 0.05)
            c03 = compute_csi(p, y, 0.30)
            ccont = compute_continuous_csi(p, y, 0.01)

            # Accumulate for summary stats
            mae_l.append(mae)
            mse_l.append(mse)
            rmse_l.append(rmse)
            rrmse_l.append(rrmse)
            c05_l.append(c05)
            c03_l.append(c03)
            ccont_l.append(ccont)
            tt_l.append(tt)

            # Persist per-sample row
            writer.writerow(
                {
                    "Sample Index": i,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "RRMSE": rrmse,
                    "CSI_0.05": c05,
                    "CSI_0.3": c03,
                    "Continuous_CSI": ccont,
                    "Test Time": tt,
                }
            )

    # --------------------------- Summary printout ---------------------------- #
    def _mean_std(v: List[float]) -> Tuple[float, float]:
        return float(np.mean(v)), float(np.std(v))

    m_mae, s_mae = _mean_std(mae_l)
    m_mse, s_mse = _mean_std(mse_l)
    m_rmse, s_rmse = _mean_std(rmse_l)
    m_rrmse, s_rrmse = _mean_std(rrmse_l)
    m_c05, s_c05 = _mean_std(c05_l)
    m_c03, s_c03 = _mean_std(c03_l)
    m_cc, s_cc = _mean_std(ccont_l)
    m_tt, s_tt = _mean_std(tt_l)

    print(f"Avg test time: {m_tt:.4f}s ± {s_tt:.4f}s")
    print(
        f"MAE {m_mae:.2e}±{s_mae:.2e} | MSE {m_mse:.2e}±{s_mse:.2e} | "
        f"RMSE {m_rmse:.2e}±{s_rmse:.2e} | RRMSE {m_rrmse:.2e}±{s_rrmse:.2e}"
    )
    print(
        f"CSI@0.05 {m_c05:.2e}±{s_c05:.2e} | CSI@0.3 {m_c03:.2e}±{s_c03:.2e} | "
        f"ContCSI {m_cc:.2e}±{s_cc:.2e}"
    )

    # ------------------------------- Plots ---------------------------------- #
    # Use one representative sample for visualization (clamped index)
    res_dir = Path(C.ckpt_path) / C.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)
    idx = min(5, len(proc) - 1)

    g = proc[idx].to(device)
    with torch.no_grad():
        pred = high_model(g.ndata["x"], g.edata["x"], g)
    pred = pred * (hi - lo) + lo
    tgt = g.ndata["y"] * (hi - lo) + lo

    def _save(arr: np.ndarray | torch.Tensor, name: str, title: str) -> None:
        """Helper: render a colored node map and save to disk."""
        fig, ax = plt.subplots(figsize=(13, 10))
        plot_mesh(ax, g, arr, "viridis", title, vmin=0, vmax=2.8)
        plt.tight_layout()
        plt.savefig(res_dir / name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Ground truth vs. prediction
    _save(tgt.detach().cpu().numpy().ravel(), "Ground_Truth.png", "Ground Truth")
    _save(pred.detach().cpu().numpy().ravel(), "Prediction.png", "Prediction")

    # Upsampled LF feature (last node feature channel in g.ndata["x"])
    stats_lf = json.loads((Path(C.ckpt_path) / C.ckpt_name_lf / "normalization_params_LF.json").read_text())
    lo_lf = torch.tensor(stats_lf["label_min"], device=device)
    hi_lf = torch.tensor(stats_lf["label_max"], device=device)
    up = g.ndata["x"][:, -1] * (hi_lf - lo_lf) + lo_lf
    _save(up.detach().cpu().numpy().ravel(), "Upsampled_Low_Fidelity.png", "Upsampled Low-Fidelity")

    # Absolute error map
    err = torch.abs(pred - tgt).detach().cpu().numpy().ravel()
    _save(err, "Error.png", "Absolute Error [m]")

    # --------------------- Node-wise error distribution --------------------- #
    # Compute per-node errors for each test graph, then average across graphs.
    node_errs = []
    for gg in proc:
        gg = gg.to(device)
        with torch.no_grad():
            pp = high_model(gg.ndata["x"], gg.edata["x"], gg)
        pp = pp * (hi - lo) + lo
        yy = gg.ndata["y"] * (hi - lo) + lo
        node_errs.append((pp - yy).detach().cpu().numpy().ravel())

    node_errs = np.stack(node_errs, axis=0)       # shape: (num_graphs, num_nodes)
    avg_node_errs = node_errs.mean(axis=0)        # mean error per node over graphs

    # Persist arrays for future analysis/replotting
    np.save(res_dir / "raw_node_errors.npy", node_errs)
    np.save(res_dir / "avg_node_errors.npy", avg_node_errs)

    # Histogram of average node-wise errors
    plt.figure(figsize=(10, 6))
    plt.hist(avg_node_errs, bins=100, edgecolor="black")
    plt.xlabel("Average Error [m]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(res_dir / "Avg_Node_Error_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Entrypoint for CLI use: runs evaluation and plotting pipeline.
    evaluate_and_plot()
