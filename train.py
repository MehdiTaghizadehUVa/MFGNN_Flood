# train.py
# Training entrypoint for LF then HF stages.

"""
Two-stage MeshGraphNet training driver (LF → HF).

Overview
--------
This script orchestrates end-to-end training for a two-stage pipeline:

1) Low-Fidelity (LF) stage
   - Trains a MeshGraphNet on coarse-resolution data (prefix `M80`).
   - Persists normalization stats and a "best" checkpoint.
   - Produces a history of train/val losses.

2) High-Fidelity (HF) stage
   - Loads the best LF model.
   - Builds HF graphs (prefix `M10`) and augments them with an interpolated
     LF prediction feature (done inside the HF trainer found in `utils.py`).
   - Trains a MeshGraphNet on augmented HF graphs.
   - Persists normalization stats and a "best" checkpoint.
   - Produces a history of train/val losses.

Distributed / Logging
---------------------
- Uses `modulus.distributed.manager.DistributedManager` for rank/device info.
- Uses `modulus.launch.logging` for rank-zero filtered logging.
- Optionally logs metrics to Weights & Biases (if `wb` is available and enabled).
- Saves periodic checkpoints (every epoch) to allow resume.

Artifacts
---------
- `<ckpt_path>/<ckpt_name_lf>/` and `<ckpt_path>/<ckpt_name_hf>/` contain
  model checkpoints and normalization JSONs.
- Loss histories are stored at `<ckpt_path>/low_fidelity_losses.pkl` and
  `<ckpt_path>/high_fidelity_losses.pkl`.

Dependencies
------------
- See `utils.py` for the trainer implementations and data-building pipeline.
- Requires a `Constants` config model (see `constants.py` or similar).
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import torch
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.launch.utils import save_checkpoint

from constants import Constants
from utils import MGNTrainerLF, MGNTrainerHF, wb  # wb is optional wandb handle


def main() -> None:
    """Run LF and HF training loops back-to-back.

    Steps
    -----
    1) Initialize distributed context (sets `rank`, `world_size`, `device`).
    2) Create checkpoint root and dump the run configuration (rank 0 only).
    3) Initialize W&B (mode controlled by `C.wandb_mode`).
    4) Train LF stage:
        - For each epoch: train → validate → log → checkpoint.
    5) Train HF stage:
        - For each epoch: train → validate → log → checkpoint.
    6) Persist train/val loss histories for both stages (rank 0 only).

    Notes
    -----
    - The underlying trainers manage data loading, normalization, and
      best-checkpoint saving based on validation loss.
    - This script saves rolling (per-epoch) checkpoints for convenience when
      resuming or inspecting intermediate states.
    """
    C = Constants()

    # Initialize distributed manager (sets CUDA device per rank and exposes rank/world_size)
    DistributedManager.initialize()
    dist = DistributedManager()

    # ----------------------------------------------------------------------------
    # Rank 0: ensure checkpoint directory exists and dump run configuration JSON
    # ----------------------------------------------------------------------------
    if dist.rank == 0:
        Path(C.ckpt_path).mkdir(parents=True, exist_ok=True)
        # Save a frozen copy of the configuration used for this run
        (Path(C.ckpt_path) / f"{C.ckpt_name_hf}.json").write_text(C.model_dump_json(indent=4))

    # ----------------------------------------------------------------------------
    # Initialize Weights & Biases (entity/project/name configurable via Constants)
    # ----------------------------------------------------------------------------
    initialize_wandb(project="Flood_GNN", entity="uva_mehdi", name="Flood-Training", mode=C.wandb_mode)

    # Base logger and rank-zero wrapper to avoid duplicate multi-GPU logs
    logger = PythonLogger("main")
    r0log = RankZeroLoggingWrapper(logger, dist)

    # =============================== LF stage ================================= #
    # Trainer encapsulates:
    # - dataset discovery, preprocessing, normalization statistics
    # - model/loss/optimizer/scheduler setup
    # - AMP support and best-checkpoint saving
    trainer_lf = MGNTrainerLF(wb, dist, r0log)

    for epoch in range(trainer_lf.epoch_init, C.epochs_LF):
        t0 = time.time()

        # Train for one epoch on normalized LF graphs
        tr = trainer_lf.train_epoch(trainer_lf.train_loader)
        # Validate (computes denormalized relative error %, saves best model internally)
        vl = trainer_lf.validate_epoch(trainer_lf.val_loader, "LF")
        # Keep loss history for later plotting/analysis
        trainer_lf.log_losses(tr, vl)

        # W&B scalar logs (only from rank 0 to avoid duplication)
        if dist.rank == 0 and wb is not None:
            wb.log({"epoch": epoch, "train_loss": tr, "val_loss": vl, "lr": trainer_lf.get_lr()})

        # Human-readable summary line
        r0log.info(
            f"[LF] Epoch {epoch} | Train {tr:.2e} | Val {vl:.2e} | "
            f"LR {trainer_lf.get_lr():.3e} | {time.time() - t0:.2f}s"
        )

        # Save rolling checkpoint each epoch (in addition to "best" saved by trainer)
        if dist.rank == 0:
            save_checkpoint(
                (Path(C.ckpt_path) / C.ckpt_name_lf).as_posix(),
                trainer_lf.model,
                trainer_lf.optimizer,
                trainer_lf.scheduler,
                trainer_lf.scaler,
                epoch,
            )

    # =============================== HF stage ================================= #
    # The HF trainer:
    # - loads best LF model
    # - builds HF graphs and appends an interpolated LF feature
    # - trains the HF MeshGraphNet on augmented graphs
    trainer_hf = MGNTrainerHF(wb, dist, r0log)

    for epoch in range(trainer_hf.epoch_init, C.epochs_HF):
        t0 = time.time()

        tr = trainer_hf.train_epoch(trainer_hf.train_loader)
        vl = trainer_hf.validate_epoch(trainer_hf.val_loader, "HF")
        trainer_hf.log_losses(tr, vl)

        if dist.rank == 0 and wb is not None:
            wb.log({"epoch": epoch, "train_loss": tr, "val_loss": vl, "lr": trainer_hf.get_lr()})

        r0log.info(
            f"[HF] Epoch {epoch} | Train {tr:.2e} | Val {vl:.2e} | "
            f"LR {trainer_hf.get_lr():.3e} | {time.time() - t0:.2f}s"
        )

        if dist.rank == 0:
            save_checkpoint(
                (Path(C.ckpt_path) / C.ckpt_name_hf).as_posix(),
                trainer_hf.model,
                trainer_hf.optimizer,
                trainer_hf.scheduler,
                trainer_hf.scaler,
                epoch,
            )

    # =========================== Save loss curves ============================= #
    # Persist loss histories for post-hoc plotting/analysis.
    if dist.rank == 0:
        with open(Path(C.ckpt_path) / "low_fidelity_losses.pkl", "wb") as f:
            pickle.dump(trainer_lf.low_fidelity_losses, f)

        with open(Path(C.ckpt_path) / "high_fidelity_losses.pkl", "wb") as f:
            pickle.dump(trainer_hf.high_fidelity_losses, f)


if __name__ == "__main__":
    # Entry point: runs both LF and HF training legs sequentially.
    main()
