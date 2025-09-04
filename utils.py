# utils.py
# Shared data loading, graph building, models, and trainer classes.

"""
Utilities for two-stage (LF/HF) MeshGraphNet training on flood datasets.

This module provides:
- File I/O and preprocessing helpers to build DGL graphs from tabular text files
- Normalization utilities and cached dataset builders for LF/HF splits
- A lightweight EdgeConv GNN (for completeness) and MeshGraphNet loaders
- Training scaffolding (base trainer + LF/HF trainers)
- LF→HF feature augmentation via spatial interpolation of LF predictions

Key concepts
------------
LF (Low Fidelity)
    Coarser-resolution simulations (prefix `M80` in file names) used to train
    an initial model and produce upsampled signals.

HF (High Fidelity)
    Finer-resolution simulations (prefix `M10`). HF graphs are augmented with
    an interpolated LF signal predicted on LF nodes, then used to train a
    second-stage model.

Conventions
-----------
- Node features `g.ndata["x"]` are float32; labels/targets `g.ndata["y"]` are
  float32, normalized to [0, 1] with min–max stats computed per-dataset.
- Edge features `g.edata["x"]` include relative Δx, Δy, and Euclidean distance.
- Normalization stats are saved to `{ckpt_dir}/normalization_params_[LF|HF].json`.
- Caching of processed graphs is optional and controlled via `cache_dir`.

Notes
-----
This module assumes the presence of a `Constants` Pydantic model (see
`constants_UP_mgn.py`) that defines paths, model dims, and training hyperparams.
"""

from __future__ import annotations
import json, os, pickle, random, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import graph as dgl_graph
from dgl.dataloading import GraphDataLoader
from dgl.nn import EdgeConv
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from constants_UP_mgn import Constants

try:
    import wandb as wb  # optional experiment tracker
except Exception:
    wb = None

# -----------------------------------------------------------------------------
# Global config / seeds
# -----------------------------------------------------------------------------
C = Constants()
random.seed(C.random_seed)
np.random.seed(C.random_seed)
torch.manual_seed(C.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(C.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Small epsilon to guard divisions against zero
EPS = 1e-8


# --------------------------- I/O & preprocessing ---------------------------- #
def _np_load_txt(path: Path) -> np.ndarray:
    """Load a tab-separated text file into a NumPy array.

    Parameters
    ----------
    path : Path
        Path to the `.txt` file.

    Returns
    -------
    np.ndarray
        Array with shape (T, D) or (N,) depending on file contents.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.loadtxt(path, delimiter="\t")


def load_constant_data(folder: str | Path, prefix: str) -> Tuple[np.ndarray, ...]:
    """Load geometry and static per-node predictors shared across scenarios.

    Expected files (prefix = 'M80' or 'M10'):
      - `{prefix}_XY.txt` : node coordinates [x, y]
      - `{prefix}_CA.txt` : contributing area
      - `{prefix}_CE.txt` : elevation
      - `{prefix}_CS.txt` : slope
      - `{prefix}_A.txt`  : aspect
      - `{prefix}_CU.txt` : curvature

    All arrays are truncated (if needed) to the number of points in XY.

    Parameters
    ----------
    folder : str | Path
        Directory containing the text files.
    prefix : str
        File prefix, e.g., `'M80'` (LF) or `'M10'` (HF).

    Returns
    -------
    tuple of np.ndarray
        `(xy, area, elev, slope, aspect, curv)`; each array has shape (N, d).
        `xy` is (N, 2); the others are (N, 1).
    """
    folder = Path(folder)
    xy = _np_load_txt(folder / f"{prefix}_XY.txt")
    n = xy.shape[0]

    def _col(name: str) -> np.ndarray:
        # Ensure column vectors and match XY length
        return _np_load_txt(folder / f"{prefix}_{name}.txt")[:n].reshape(-1, 1)

    area = _col("CA")
    elev = _col("CE")
    slope = _col("CS")
    aspect = _col("A")
    curv = _col("CU")
    return xy, area, elev, slope, aspect, curv


def load_static_data(
    folder: str | Path,
    hid: str,
    prefix: str,
    num_pts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-scenario dynamic fields and derive hydrograph features.

    Reads:
      - Water depth time series per node: `{prefix}_WD_{hid}.txt`
      - Velocity components time series:  `{prefix}_VX_{hid}.txt`, `{prefix}_VY_{hid}.txt`
      - Upstream inflow hydrograph:      `{prefix}_US_InF_{hid}.txt` (we use column 2)

    From inflow, compute compact features (position/shape/intensity/duration).

    Parameters
    ----------
    folder : str | Path
        Directory containing the scenario files.
    hid : str
        Hydrograph/scenario identifier embedded in file names.
    prefix : str
        File prefix, e.g., `'M80'` or `'M10'`.
    num_pts : int
        Number of spatial points to keep per time series (truncates if needed).

    Returns
    -------
    max_wd : np.ndarray
        Maximum water depth per node over time, shape (N,).
    max_vm : np.ndarray
        Maximum velocity magnitude per node over time, shape (N,). (Computed
        but not used downstream; returned for completeness.)
    h_feats : np.ndarray
        Vector of hydrograph features, shape (9,): [rp, rcg, m1, m2, m3, m5, ni, Ptot, dur].
    """
    folder = Path(folder)
    wd = _np_load_txt(folder / f"{prefix}_WD_{hid}.txt")[:, :num_pts]
    vx = _np_load_txt(folder / f"{prefix}_VX_{hid}.txt")[:, :num_pts]
    vy = _np_load_txt(folder / f"{prefix}_VY_{hid}.txt")[:, :num_pts]
    inflow = _np_load_txt(folder / f"{prefix}_US_InF_{hid}.txt")[:, 1]

    # Node-wise maxima across time
    max_wd = np.max(wd, axis=0)
    max_vm = np.sqrt(np.max(vx**2 + vy**2, axis=0))  # velocity magnitude (unused downstream)

    # Compact hydrograph descriptors
    idx_peak = int(np.argmax(inflow))
    rp = idx_peak / len(inflow)  # relative position of peak [0..1]
    csum = np.cumsum(inflow)
    rcg = int(np.searchsorted(csum, 0.5 * csum[-1])) / len(inflow)  # center of gravity (cum.)
    m1 = np.sum(inflow[:idx_peak]) / (np.sum(inflow[idx_peak:]) + EPS)  # pre/post-peak mass balance
    m2 = np.max(inflow) / (np.sum(inflow) + EPS)  # peak-to-total ratio
    m3 = np.sum(inflow[: len(inflow)//3]) / (np.sum(inflow) + EPS)  # first-third mass ratio
    m5 = np.sum(inflow[: len(inflow)//2]) / (np.sum(inflow) + EPS)  # first-half mass ratio
    ni = np.max(inflow) / (np.mean(inflow) + EPS)  # normalized intensity
    Ptot = np.sum(inflow)                           # total volume proxy
    dur = len(inflow)                               # event duration [timesteps]

    h_feats = np.array([rp, rcg, m1, m2, m3, m5, ni, Ptot, dur])
    return max_wd, max_vm, h_feats


def create_static_node_features(
    xy: np.ndarray,
    area: np.ndarray,
    elev: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    curv: np.ndarray,
    h_feats: np.ndarray,
) -> np.ndarray:
    """Assemble per-node features by concatenating static grids and hydrograph stats.

    Features layout:
      [x, y, area, elev, slope, aspect, curvature, *hydrograph_features]

    Hydrograph features are repeated per node (global per-scenario scalars).

    Returns
    -------
    np.ndarray
        Feature matrix with shape (N, D).
    """
    # Repeat scenario-level features at each node
    h_rep = np.tile(h_feats, (xy.shape[0], 1))
    return np.hstack([xy, area, elev, slope, aspect, curv, h_rep])


def create_edge_features(xy: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Compute edge features given node coordinates and an edge index.

    Edge features are:
      - Δx, Δy between source and destination nodes
      - Euclidean distance ||Δ||

    Parameters
    ----------
    xy : np.ndarray
        Node coordinates, shape (N, 2).
    edge_index : np.ndarray
        Array of shape (2, E) with src/dst node indices.

    Returns
    -------
    np.ndarray
        Edge feature matrix of shape (E, 3).
    """
    row, col = edge_index
    rel = xy[row] - xy[col]
    dist = np.linalg.norm(rel, axis=1)
    return np.hstack([rel, dist[:, None]])


def build_static_graph(
    xy: np.ndarray,
    node_feats: np.ndarray,
    edge_feats: np.ndarray,
    targets: np.ndarray,
    edge_index: np.ndarray,
) -> dgl_graph:
    """Create a DGL graph with node/edge features and per-node labels.

    Parameters
    ----------
    xy : np.ndarray
        Node coordinates (unused directly; helpful for debugging/plots).
    node_feats : np.ndarray
        Node feature matrix (N, Dn).
    edge_feats : np.ndarray
        Edge feature matrix (E, De).
    targets : np.ndarray
        Per-node labels (N, Dy), typically normalized water depth.
    edge_index : np.ndarray
        Array of shape (2, E) with src/dst indices.

    Returns
    -------
    dgl.DGLGraph
        Graph with fields:
          - g.ndata["x"] : float32 node features
          - g.ndata["y"] : float32 labels
          - g.edata["x"] : float32 edge features
    """
    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata["x"] = torch.tensor(node_feats, dtype=torch.float32)
    g.edata["x"] = torch.tensor(edge_feats, dtype=torch.float32)
    g.ndata["y"] = torch.tensor(targets, dtype=torch.float32)
    return g


def _k_nn_edges(xy: np.ndarray, k: int) -> np.ndarray:
    """Build a directed k-NN edge list (without self loops).

    For each node i, connect i → each of its k nearest neighbors.

    Parameters
    ----------
    xy : np.ndarray
        Node coordinates (N, 2).
    k : int
        Number of neighbors per node.

    Returns
    -------
    np.ndarray
        Edge index of shape (2, E), dtype int64.
    """
    # KDTree query returns (distances, indices); query includes the node itself.
    nbrs = KDTree(xy).query(xy, k=k+1)[1]
    pairs = [(i, j) for i, row in enumerate(nbrs) for j in row if j != i]
    return np.array(pairs, dtype=np.int64).T


def _minmax_accum(a_min: np.ndarray, a_max: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate elementwise minima and maxima for streaming min–max stats."""
    return np.minimum(a_min, x.min(axis=0)), np.maximum(a_max, x.max(axis=0))


def _minmax_norm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Min–max normalize `x` using bounds `lo`/`hi` with epsilon protection."""
    return (x - lo) / (hi - lo + EPS)


def process_static_folder(
    folder: str | Path,
    hydrograph_ids: Iterable[str],
    is_low_fidelity: bool = True,
    k: int = 4,
    cache_dir: Optional[str | Path] = None,
) -> Tuple[Dict[str, dgl_graph], Dict[str, List[float]]]:
    """Load, normalize, and (optionally) cache graphs for a set of scenarios.

    Two-pass procedure:
      1) Pass 1 computes min–max statistics over all selected scenarios.
      2) Pass 2 normalizes node/edge/label tensors and builds DGL graphs.

    A pickle cache (if `cache_dir` provided) stores `(graphs, stats)`.

    Parameters
    ----------
    folder : str | Path
        Directory with input text files (XY, CA, CE, ... and WD/VX/VY/US_InF).
    hydrograph_ids : Iterable[str]
        Scenario IDs to include.
    is_low_fidelity : bool, default True
        Uses `'M80'` if True, `'M10'` otherwise.
    k : int, default 4
        k-NN edges per node.
    cache_dir : str | Path | None
        If set, enables load/save of a cached pickle.

    Returns
    -------
    graphs : Dict[str, dgl.DGLGraph]
        Mapping `hid → graph` (normalized).
    stats : Dict[str, List[float]]
        Min–max stats for features, labels, and edge features.
    """
    folder = Path(folder)
    prefix = "M80" if is_low_fidelity else "M10"
    ids = list(hydrograph_ids)

    # Try cache first (fast path)
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{prefix}_graphs_{len(ids)}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                return pickle.load(f)

    # Common geometry/static grids
    xy, area, elev, slope, aspect, curv = load_constant_data(folder, prefix)
    edge_index = _k_nn_edges(xy, k=k)
    edge_feats = create_edge_features(xy, edge_index)

    graphs: Dict[str, dgl_graph] = {}
    # Initialize extrema with +/- inf (vectorized, dim matches features)
    fmin = np.full((xy.shape[1] + area.shape[1] + elev.shape[1] + slope.shape[1] + aspect.shape[1] + curv.shape[1] + 9,), np.inf)
    fmax = -fmin
    ymin = np.array([np.inf])
    ymax = -ymin
    emin = np.full((edge_feats.shape[1],), np.inf)
    emax = -emin

    cached_nodes_targets: List[Tuple[np.ndarray, np.ndarray]] = []

    # --------------------------- Pass 1: extrema ---------------------------- #
    for hid in tqdm(ids, desc="Pass1: extrema"):
        wd_max, _, h_feats = load_static_data(folder, hid, prefix, xy.shape[0])
        x_node = create_static_node_features(xy, area, elev, slope, aspect, curv, h_feats)
        y = wd_max.reshape(-1, 1)

        cached_nodes_targets.append((x_node, y))
        fmin, fmax = _minmax_accum(fmin, fmax, x_node)
        ymin, ymax = _minmax_accum(ymin, ymax, y)
        emin, emax = _minmax_accum(emin, emax, edge_feats)

    # --------------------------- Pass 2: build ------------------------------ #
    for hid, (x_node, y) in tqdm(list(zip(ids, cached_nodes_targets)), total=len(ids), desc="Pass2: build"):
        x_node_n = _minmax_norm(x_node, fmin, fmax)
        y_n = _minmax_norm(y, ymin, ymax)
        e_n = _minmax_norm(edge_feats, emin, emax)
        graphs[hid] = build_static_graph(xy, x_node_n, e_n, y_n, edge_index)

    stats = {
        "feature_min": fmin.tolist(),
        "feature_max": fmax.tolist(),
        "label_min":   ymin.tolist(),
        "label_max":   ymax.tolist(),
        "edge_feature_min": emin.tolist(),
        "edge_feature_max": emax.tolist(),
    }

    # Save cache for re-use
    if cache_file:
        with cache_file.open("wb") as f:
            pickle.dump((graphs, stats), f)

    return graphs, stats


def process_test_folder(
    folder: str | Path,
    hydrograph_ids: Iterable[str],
    is_low_fidelity: bool = True,
    k: int = 4,
    cache_dir: Optional[str | Path] = None,
) -> Dict[str, dgl_graph]:
    """Build normalized test graphs using previously-saved normalization stats.

    This function does **not** recompute min–max stats; it loads them from
    `{ckpt_dir}/normalization_params_[LF|HF].json` and applies them.

    Parameters
    ----------
    folder : str | Path
        Directory with input text files (XY, CA, CE, ... and WD/VX/VY/US_InF).
    hydrograph_ids : Iterable[str]
        Scenario IDs to include.
    is_low_fidelity : bool, default True
        Uses `'M80'` if True, `'M10'` otherwise; determines which stats to load.
    k : int, default 4
        k-NN edges per node.
    cache_dir : str | Path | None
        Optional pickle cache for speed.

    Returns
    -------
    Dict[str, dgl.DGLGraph]
        Mapping `hid → graph` (normalized using saved stats).
    """
    folder = Path(folder)
    prefix = "M80" if is_low_fidelity else "M10"
    tag = "LF" if is_low_fidelity else "HF"
    ids = list(hydrograph_ids)

    # Try cache
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{prefix}_test_graphs_{len(ids)}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                return pickle.load(f)

    # Load stats from the appropriate checkpoint directory
    stat_dir = Path(C.ckpt_path) / (C.ckpt_name_lf if is_low_fidelity else C.ckpt_name_hf)
    stats = json.loads((stat_dir / f"normalization_params_{tag}.json").read_text())

    fmin = np.array(stats["feature_min"])
    fmax = np.array(stats["feature_max"])
    ymin = np.array(stats["label_min"])
    ymax = np.array(stats["label_max"])
    emin = np.array(stats["edge_feature_min"])
    emax = np.array(stats["edge_feature_max"])

    xy, area, elev, slope, aspect, curv = load_constant_data(folder, prefix)
    edge_index = _k_nn_edges(xy, k=k)

    out: Dict[str, dgl_graph] = {}
    for hid in tqdm(ids, desc="Build test graphs"):
        wd_max, _, h_feats = load_static_data(folder, hid, prefix, xy.shape[0])
        x_node = create_static_node_features(xy, area, elev, slope, aspect, curv, h_feats)
        e_feats = create_edge_features(xy, edge_index)

        # Apply saved normalization
        x_node_n = _minmax_norm(x_node, fmin, fmax)
        y_n = _minmax_norm(wd_max.reshape(-1, 1), ymin, ymax)
        e_n = _minmax_norm(e_feats, emin, emax)

        out[hid] = build_static_graph(xy, x_node_n, e_n, y_n, edge_index)

    # Save cache
    if cache_file:
        with cache_file.open("wb") as f:
            pickle.dump(out, f)

    return out


def process_low_fidelity_folder(
    folder: str | Path,
    hydrograph_ids: Iterable[str],
    k: int = 4,
    cache_dir: Optional[str | Path] = None,
):
    """Thin wrapper over `process_static_folder(..., is_low_fidelity=True)`."""
    return process_static_folder(folder, hydrograph_ids, is_low_fidelity=True, k=k, cache_dir=cache_dir)


def process_high_fidelity_folder(
    folder: str | Path,
    hydrograph_ids: Iterable[str],
    k: int = 4,
    cache_dir: Optional[str | Path] = None,
):
    """Thin wrapper over `process_static_folder(..., is_low_fidelity=False)`."""
    return process_static_folder(folder, hydrograph_ids, is_low_fidelity=False, k=k, cache_dir=cache_dir)


# --------------------------------- Models ----------------------------------- #
class GNN(nn.Module):
    """Simple EdgeConv-based regressor.

    Architecture
    ------------
    - 2-layer MLP encoder on node features
    - `num_gnn_layers` EdgeConv layers with ReLU
    - 2-layer MLP decoder to scalar output
    - Optional residual connection (adds last input channel)
    - Hard thresholding: outputs below `threshold` are clamped to zero

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Hidden size used throughout encoder/GNN/decoder.
    output_dim : int
        Output dimension (usually 1).
    num_gnn_layers : int
        Number of EdgeConv layers.
    residual : bool, default False
        If True, add the last feature channel to the decoder output.
    threshold : float, default 0.01
        Values below this are set to zero (sparsity/denoise heuristic).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_gnn_layers: int, residual: bool = False, threshold: float = 0.01) -> None:
        super().__init__()
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            EdgeConv(in_feat=hidden_dim, out_feat=hidden_dim, batch_norm=False, allow_zero_in_degree=True)
            for _ in range(num_gnn_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim),
        )
        self.residual = residual
        self.threshold = threshold

    def forward(self, g: dgl_graph, feats: torch.Tensor) -> torch.Tensor:
        """Run forward pass on a DGL graph.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph (unused by MLP blocks; used by EdgeConv).
        feats : torch.Tensor
            Node features of shape (N, D_in).

        Returns
        -------
        torch.Tensor
            Predictions per node, shape (N, D_out).
        """
        # Optional residual: use last input channel as an additive bias
        x_res = feats[:, -1].view(-1, 1) if self.residual else 0.0

        # Encode → GNN → Decode
        x = F.relu(self.encoder1(feats))
        x = F.relu(self.encoder2(x))
        for layer in self.gnn_layers:
            x = F.relu(layer(g, x))
        out = self.decoder(x) + x_res

        # Clamp very small values to zero (heuristic threshold)
        return torch.where(out < self.threshold, torch.tensor(0.0, device=out.device), out)


def load_model(model_path: str | Path, device: torch.device, model_type: str):
    """Instantiate and load a model checkpoint.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model weights (.mdl/us or .pt depending on model).
    device : torch.device
        Device to place the model on.
    model_type : str
        One of {"MeshGraphNet_LF", "MeshGraphNet_HF", "GNN"}.

    Returns
    -------
    torch.nn.Module
        A ready-to-eval model instance on `device`.

    Raises
    ------
    ValueError
        If `model_type` is unsupported.
    """
    model_path = Path(model_path)

    if model_type == "MeshGraphNet_LF":
        model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation,
            processor_size=C.processor_size, hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder, hidden_dim_node_decoder=C.hidden_dim_node_decoder,
            hidden_dim_processor=C.hidden_dim_processor, threshold=C.threshold,
        ).to(device)
        model.load(model_path.as_posix())

    elif model_type == "MeshGraphNet_HF":
        model = MeshGraphNet(
            C.input_dim_nodes + 1, C.input_dim_edges, C.output_dim, aggregation=C.aggregation,
            processor_size=C.processor_size, hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder, hidden_dim_node_decoder=C.hidden_dim_node_decoder,
            hidden_dim_processor=C.hidden_dim_processor, use_residual=C.use_residual, threshold=C.threshold_hf,
        ).to(device)
        model.load(model_path.as_posix())

    elif model_type == "GNN":
        model = GNN(C.input_dim, C.hidden_dim, C.output_dim, C.num_gnn_layers, residual=C.use_residual).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        raise ValueError("Unsupported model type")

    model.eval()
    return model


# ------------------------------ Training base -------------------------------- #
class BaseTrainer:
    """Common training/validation scaffold shared by LF/HF trainers.

    Responsibilities
    ----------------
    - Mixed precision (AMP) support via `GradScaler` (controlled by `C.amp`)
    - Logging hooks (W&B optional) and rank-zero logging
    - LR scheduling and best-model checkpointing on validation loss
    - Denormalization helper to compute scale-aware validation errors

    Subclasses must:
    - Define `self.model`, `self.optimizer`, `self.scheduler`
    - Implement `_save_best_model(self, tag: str) -> None`
    - Populate `self.normalization_params` with label min/max for denorm
    """

    def __init__(self, wandb_mod, dist: DistributedManager, logger: RankZeroLoggingWrapper) -> None:
        self.wb = wandb_mod
        self.dist = dist
        self.rank_zero_logger = logger

        self.scaler = GradScaler()
        self.best_val_loss = float("inf")
        self.epoch_init = 0
        self.normalization_params: Dict[str, List[float]] = {}

        # The following are set in subclasses
        self.model = None
        self.optimizer = None
        self.scheduler = None

    @staticmethod
    def _denorm(x: torch.Tensor, lo: np.ndarray | torch.Tensor, hi: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Min–max denormalize a tensor using numpy or torch bounds."""
        lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
        hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
        return x * (hi - lo) + lo

    def get_lr(self) -> float:
        """Return the current learning rate from the first param group."""
        for g in self.optimizer.param_groups:
            return float(g["lr"])
        return float("nan")

    def _step_batch(self, graph: dgl_graph) -> torch.Tensor:
        """Compute the scale-invariant loss for a single mini-batch.

        Loss
        ----
        L = ||pred - y||_2 / (||y||_2 + eps)
        """
        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            diff = torch.flatten(pred) - torch.flatten(graph.ndata["y"])
            loss = torch.norm(diff, p=2) / (torch.norm(torch.flatten(graph.ndata["y"]), p=2) + EPS)
        return loss

    def train_epoch(self, loader: GraphDataLoader) -> float:
        """Run one training epoch over `loader`, applying AMP if enabled."""
        self.model.train()
        total = 0.0
        for g in loader:
            g = g.to(self.dist.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step_batch(g)

            if C.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total += loss.item()

        self.scheduler.step()
        return total / max(1, len(loader))

    @torch.no_grad()
    def validate_epoch(self, loader: GraphDataLoader, tag: str) -> float:
        """Evaluate on `loader`, log % error, and save best checkpoint.

        Returns
        -------
        float
            Mean validation loss (MSE on normalized targets).
        """
        self.model.eval()
        err_acc = 0.0
        loss_acc = 0.0

        # Pull label min/max for denormalization
        lab_lo = np.array(self.normalization_params["label_min"])
        lab_hi = np.array(self.normalization_params["label_max"])

        for g in loader:
            g = g.to(self.dist.device)

            # Forward on normalized tensors
            pred = self.model(g.ndata["x"], g.edata["x"], g)

            # Denormalize to compute scale-aware error
            pred_de = self._denorm(pred, lab_lo, lab_hi)
            gt = self._denorm(g.ndata["y"], lab_lo, lab_hi)

            # Relative L2 error (percent)
            err = torch.mean(torch.norm(pred_de - gt, p=2) / (torch.norm(gt, p=2) + EPS)).item()
            loss = torch.mean((pred - g.ndata["y"]) ** 2).item()  # MSE on normalized range

            err_acc += err
            loss_acc += loss

        err_pct = (err_acc / max(1, len(loader))) * 100.0
        if self.wb is not None:
            self.wb.log({f"{tag}_val_error (%)": err_pct})
        self.rank_zero_logger.info(f"[{tag}] validation error (%): {err_pct:.3f}")

        avg_loss = loss_acc / max(1, len(loader))

        # Checkpoint best model (rank 0 only)
        if avg_loss < self.best_val_loss and self.dist.rank == 0:
            self.best_val_loss = avg_loss
            self._save_best_model(tag)

        return avg_loss

    def _save_best_model(self, tag: str) -> None:
        """Subclass hook: save best model artifacts."""
        raise NotImplementedError


# --------------------------- LF / HF trainers -------------------------------- #
class MGNTrainerLF(BaseTrainer):
    """Low-fidelity trainer (LF, prefix `M80`).

    Loads random LF scenarios, computes/saves normalization stats, builds
    dataloaders, initializes MeshGraphNet-LF, and trains/validates.
    """

    def __init__(self, wandb_mod, dist: DistributedManager, logger: RankZeroLoggingWrapper) -> None:
        super().__init__(wandb_mod, dist, logger)
        self.low_fidelity_losses = {"train": [], "val": []}
        self._load_dataset()
        self._init_model()

    def _load_dataset(self) -> None:
        """Discover LF IDs, preprocess graphs, persist stats, split data."""
        self.rank_zero_logger.info("[LF] Loading dataset...")

        # Derive scenario IDs from files named like "M80_WD_<ID>.txt"
        all_ids = sorted([Path(f).stem.split("_")[2] for f in os.listdir(C.data_dir) if f.startswith("M80_WD_")])
        ids = random.sample(all_ids, C.N_LF)

        graphs, stats = process_low_fidelity_folder(C.data_dir, ids, cache_dir=C.cache_dir)
        self.normalization_params = stats

        # Save stats for later test-time normalization
        lf_dir = Path(C.ckpt_path) / C.ckpt_name_lf
        lf_dir.mkdir(parents=True, exist_ok=True)
        (lf_dir / "normalization_params_LF.json").write_text(json.dumps(stats))

        # Train/val split and dataloaders
        data = list(graphs.values())
        self.train_set, self.val_set = train_test_split(data, test_size=0.1, random_state=C.random_seed)
        self.train_loader = GraphDataLoader(self.train_set, batch_size=C.batch_size, shuffle=True, drop_last=False,
                                            pin_memory=True, use_ddp=self.dist.world_size > 1)
        self.val_loader = GraphDataLoader(self.val_set, batch_size=C.batch_size, shuffle=False, drop_last=False,
                                          pin_memory=True, use_ddp=False)

    def _init_model(self) -> None:
        """Instantiate MeshGraphNet-LF and restore checkpoint if present."""
        self.model = MeshGraphNet(
            C.input_dim_nodes, C.input_dim_edges, C.output_dim, aggregation=C.aggregation,
            processor_size=C.processor_size, hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder, hidden_dim_node_decoder=C.hidden_dim_node_decoder,
            hidden_dim_processor=C.hidden_dim_processor, threshold=C.threshold,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.dist.local_rank], output_device=self.dist.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda ep: C.lr_decay_rate ** ep)

        # Restore (if any) and set starting epoch
        self.epoch_init = load_checkpoint(
            (Path(C.ckpt_path) / C.ckpt_name_lf).as_posix(),
            models=self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, device=self.dist.device
        )

    def _save_best_model(self, tag: str) -> None:
        """Persist the best LF checkpoint (rank-zero only)."""
        outdir = Path(C.ckpt_path) / C.ckpt_name_lf / "best_model_LF"
        save_checkpoint(outdir.as_posix(), models=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, scaler=self.scaler, epoch=0)
        self.rank_zero_logger.info(f"[LF] Best model saved (val loss: {self.best_val_loss:.3e})")

    def log_losses(self, tr: float, vl: float) -> None:
        """Append training/validation losses to history (for plotting later)."""
        self.low_fidelity_losses["train"].append(tr)
        self.low_fidelity_losses["val"].append(vl)


class MGNTrainerHF(BaseTrainer):
    """High-fidelity trainer (HF, prefix `M10`).

    Workflow
    --------
    - Load best LF model
    - Build LF/HF graphs for the same scenario IDs
    - Interpolate LF predictions onto HF nodes (append as last feature)
    - Train MeshGraphNet-HF on augmented HF graphs
    """

    def __init__(self, wandb_mod, dist: DistributedManager, logger: RankZeroLoggingWrapper) -> None:
        super().__init__(wandb_mod, dist, logger)
        self.high_fidelity_losses = {"train": [], "val": []}
        self._load_dataset()
        self._init_model()

    def _load_dataset(self) -> None:
        """Prepare HF graphs augmented with interpolated LF predictions."""
        self.rank_zero_logger.info("[HF] Loading dataset...")

        # Load best LF for upsampling signal
        lf_best = Path(C.ckpt_path) / C.ckpt_name_lf / "best_model_LF" / "MeshGraphNet.0.0.mdlus"
        low_model = load_model(lf_best, self.dist.device, "MeshGraphNet_LF")

        # Use random subset of HF scenarios
        all_ids = sorted([Path(f).stem.split("_")[2] for f in os.listdir(C.data_dir) if f.startswith("M10_WD_")])
        ids_tr = sorted(random.sample(all_ids, C.N_HF))

        # Build normalized test-style graphs for LF/HF
        lf_graphs = process_test_folder(C.data_dir, ids_tr, is_low_fidelity=True, cache_dir=C.cache_dir)
        hf_graphs, stats = process_high_fidelity_folder(C.data_dir, ids_tr, cache_dir=C.cache_dir)
        self.normalization_params = stats

        # Save HF stats
        hf_dir = Path(C.ckpt_path) / C.ckpt_name_hf
        hf_dir.mkdir(parents=True, exist_ok=True)
        (hf_dir / "normalization_params_HF.json").write_text(json.dumps(stats))

        # Append interpolated LF signal as final HF node feature
        updated = update_data(list(hf_graphs.values()), list(lf_graphs.values()), low_model, self.dist.device)

        # Split and loaders
        self.train_set, self.val_set = train_test_split(updated, test_size=0.33, random_state=C.random_seed)
        self.train_loader = GraphDataLoader(self.train_set, batch_size=2, shuffle=True)
        self.val_loader = GraphDataLoader(self.val_set, batch_size=2, shuffle=False)

    def _init_model(self) -> None:
        """Instantiate MeshGraphNet-HF and restore checkpoint if present."""
        self.model = MeshGraphNet(
            C.input_dim_nodes + 1, C.input_dim_edges, C.output_dim, aggregation=C.aggregation,
            processor_size=C.processor_size, hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder, hidden_dim_node_decoder=C.hidden_dim_node_decoder,
            hidden_dim_processor=C.hidden_dim_processor, use_residual=C.use_residual, threshold=C.threshold_hf,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[self.dist.local_rank], output_device=self.dist.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr_hf, weight_decay=C.weight_decay_hf)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda ep: C.lr_decay_rate_hf ** ep)

        self.epoch_init = load_checkpoint(
            (Path(C.ckpt_path) / C.ckpt_name_hf).as_posix(),
            models=self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, device=self.dist.device
        )

    def _save_best_model(self, tag: str) -> None:
        """Persist the best HF checkpoint (rank-zero only)."""
        outdir = Path(C.ckpt_path) / C.ckpt_name_hf / "best_model_HF"
        save_checkpoint(outdir.as_posix(), models=self.model, optimizer=self.optimizer,
                        scheduler=self.scheduler, scaler=self.scaler, epoch=0)
        self.rank_zero_logger.info(f"[HF] Best model saved (val loss: {self.best_val_loss:.3e})")

    def log_losses(self, tr: float, vl: float) -> None:
        """Append training/validation losses to history (for plotting later)."""
        self.high_fidelity_losses["train"].append(tr)
        self.high_fidelity_losses["val"].append(vl)


# ------------------------- LF->HF feature augmentation ----------------------- #
@torch.no_grad()
def update_data(
    hf_graphs: List[dgl_graph],
    lf_graphs: List[dgl_graph],
    low_model: MeshGraphNet,
    device: torch.device,
    method: str = "kn",
    k_neighbors: int = C.k_neighbors,
    epsilon: float = 5000.0,
) -> List[dgl_graph]:
    """Append an upsampled LF scalar feature to HF node features.

    For each HF graph, interpolate the LF model's node-wise prediction
    (computed on the corresponding LF graph) onto HF nodes using either:
    - "kn": k-nearest neighbors with inverse-distance weights
    - "rbf": radial basis function weights exp(-epsilon * d^2)

    The resulting scalar field is concatenated as the **last** channel in
    `hg.ndata["x"]`.

    Parameters
    ----------
    hf_graphs : list[dgl.DGLGraph]
        High-fidelity graphs (targets and base features already normalized).
    lf_graphs : list[dgl.DGLGraph]
        Corresponding low-fidelity graphs (same ordering as HF).
    low_model : MeshGraphNet
        Trained LF model used to produce the upsampled signal.
    device : torch.device
        Device to run inference/interpolation on.
    method : {"kn", "rbf"}, default "kn"
        Interpolation scheme.
    k_neighbors : int, default C.k_neighbors
        Number of neighbors for "kn" scheme (capped by LF node count).
    epsilon : float, default 5000.0
        RBF sharpness; larger → more local weighting.

    Returns
    -------
    list[dgl.DGLGraph]
        Augmented HF graphs with an extra node feature channel.

    Notes
    -----
    - Uses Euclidean distances in (x, y) space from the **first two** node
      feature columns: `[:, :2]` must be coordinates.
    - LF predictions are computed once per LF graph and then interpolated.
    """
    low_model.eval()
    total_t = 0.0
    out: List[dgl_graph] = []

    for hg, lg in zip(hf_graphs, lf_graphs):
        t0 = time.time()
        lg = lg.to(device)
        hg = hg.to(device)

        # 1) Predict LF scalar on LF nodes
        low_out = low_model(lg.ndata["x"], lg.edata["x"], lg).detach()  # [Nl, 1]

        # 2) Distances between HF and LF coordinates (first two features are x,y)
        dists = torch.cdist(hg.ndata["x"][:, :2], lg.ndata["x"][:, :2])  # [Nh, Nl]

        # 3) Interpolate LF scalar onto HF nodes
        if method == "kn":
            k = min(k_neighbors, dists.shape[1])
            d, idx = dists.topk(k, largest=False)           # nearest k LF nodes
            w = 1.0 / (d + EPS)                             # inverse distance
            w = w / (torch.sum(w, dim=1, keepdim=True) + EPS)
            interp = torch.sum(low_out[idx] * w.unsqueeze(-1), dim=1)  # [Nh, 1]

        elif method == "rbf":
            w = torch.exp(-epsilon * (dists ** 2))
            w = w / (torch.sum(w, dim=1, keepdim=True) + EPS)
            interp = torch.mm(w, low_out)                   # [Nh, 1]

        else:
            raise ValueError("method must be 'kn' or 'rbf'")

        # 4) Append as final node feature
        hg.ndata["x"] = torch.cat([hg.ndata["x"], interp], dim=1)
        out.append(hg)

        total_t += (time.time() - t0)

    print(f"Avg update_data time: {total_t / max(1, len(hf_graphs)):.4f}s")
    return out
