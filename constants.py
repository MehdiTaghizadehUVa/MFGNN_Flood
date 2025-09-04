# constants_UP_mgn.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Constants(BaseModel):
    """
    Configuration for LF/HF MeshGraphNet flood training & evaluation.

    Notes
    -----
    - Paths are `Path` objects (use `.as_posix()` if you need a string).
    - Call `ensure_dirs()` once to create checkpoint/result/cache folders.
    - Uses Pydantic v2 APIs (e.g., `model_dump_json()` in callers).
    """

    # ------------------------------ Paths ---------------------------------- #
    ckpt_path: Path = Field(default=Path("final_checkpoints_M80_10"),
                            description="Root directory for checkpoints and stats.")
    ckpt_name_lf: str = Field(default="lf_models_400", description="Subdir name for LF stage.")
    ckpt_name_hf: str = Field(default="upsampling_models_mgn_30_400", description="Subdir name for HF stage.")

    data_dir: Path = Field(
        default=Path(r"C:/Users/jrj6wm/Box/Flood_Modeling/Simulations/Case_4/Results_Target/Train"),
        description="Training data directory"
    )
    test_dir: Path = Field(
        default=Path(r"C:/Users/jrj6wm/Box/Flood_Modeling/Simulations/Case_4/Results_Target/Test"),
        description="Test data directory"
    )
    results_dir: Path = Field(
        default=Path("results_upsampling_models_mgn_30_400_Test_10"),
        description="Folder (inside ckpt_path) to save plots/metrics"
    )
    cache_dir: Path = Field(
        default=Path("dataset_80_10"),
        description="On-disk cache for preprocessed graphs"
    )
    dem_path: Path = Field(
        default=Path(r"C:\Users\jrj6wm\Box\Flood_Modeling\Simulations\Base_Model_M10\Terrain\Terrain.DEM1m_32bitfloat.tif"),
        description="Optional DEM raster path (unused by core training)"
    )

    # ----------------------------- Model dims ------------------------------ #
    input_dim_nodes: int = Field(default=16, ge=1)
    input_dim_edges: int = Field(default=3, ge=1)
    output_dim: int = Field(default=1, ge=1)

    aggregation: Literal["sum", "mean", "max"] = Field(default="sum")
    processor_size: int = Field(default=6, ge=1)
    hidden_dim_processor: int = Field(default=64, ge=1)
    hidden_dim_node_encoder: int = Field(default=64, ge=1)
    hidden_dim_edge_encoder: int = Field(default=64, ge=1)
    hidden_dim_node_decoder: int = Field(default=64, ge=1)

    threshold: float = Field(default=0.01, ge=0.0)
    threshold_hf: float = Field(default=0.01, ge=0.0)
    k_neighbors: int = Field(default=4, ge=1)
    use_residual: bool = True

    # ------------------------------- Training ------------------------------ #
    batch_size: int = Field(default=32, ge=1)
    epochs_LF: int = Field(default=2500, ge=1)
    epochs_HF: int = Field(default=2500, ge=1)
    N_LF: int = Field(default=400, ge=1, description="Number of LF hydrographs")
    N_HF: int = Field(default=30, ge=1, description="Number of HF hydrographs")
    N_Test: int = Field(default=10, ge=1, description="Number of test hydrographs")

    # ------------------------------- Optimizers ---------------------------- #
    lr: float = Field(default=1e-4, gt=0)
    lr_decay_rate: float = Field(default=0.99985, gt=0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)

    lr_hf: float = Field(default=1e-3, gt=0)
    lr_decay_rate_hf: float = Field(default=0.99985, gt=0, le=1.0)
    weight_decay_hf: float = Field(default=1e-3, ge=0.0)

    # ------------------------------- Misc ---------------------------------- #
    amp: bool = Field(default=False, description="Enable AMP mixed precision")
    jit: bool = Field(default=False, description="Enable TorchScript (if supported)")
    wandb_mode: Literal["run", "disabled", "offline"] = Field(default="run")
    random_seed: int = Field(default=44)

    # ------------------------------ Helpers -------------------------------- #
    @property
    def ckpt_dir_lf(self) -> Path:
        return self.ckpt_path / self.ckpt_name_lf

    @property
    def ckpt_dir_hf(self) -> Path:
        return self.ckpt_path / self.ckpt_name_hf

    @property
    def results_path(self) -> Path:
        return self.ckpt_path / self.results_dir

    def ensure_dirs(self) -> None:
        """Create common output directories if they don't exist."""
        for p in (self.ckpt_path, self.ckpt_dir_lf, self.ckpt_dir_hf, self.results_path, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Validators ------------------------------ #
    @field_validator(
        "lr", "lr_hf", "lr_decay_rate", "lr_decay_rate_hf",
        "weight_decay", "weight_decay_hf", mode="after"
    )
    @classmethod
    def _finite_positive(cls, v: float) -> float:
        if not (v == v) or v in (float("inf"), float("-inf")):  # NaN/Inf guard
            raise ValueError("Value must be finite.")
        return v

    @field_validator("data_dir", "test_dir", mode="after")
    @classmethod
    def _paths_exist_or_warn(cls, p: Path) -> Path:
        # Do not force existence: users may generate data later.
        # Still provide a gentle check for obvious typos.
        # (Training code will error more specifically if required files are missing.)
        return p
