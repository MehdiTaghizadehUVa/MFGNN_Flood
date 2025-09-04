# Multi-Fidelity Graph Neural Networks (MFGNN) for Flood Hazard Mapping

> Reference implementation accompanying the paper **“Multi-fidelity graph neural networks for efficient and accurate flood hazard mapping.”** Environmental Modelling & Software, 193 (2025) 106654. DOI: 10.1016/j.envsoft.2025.106654.  fileciteturn0file0

This repository provides code to train and evaluate a **two-stage, multi-fidelity GNN (MFGNN)** that combines many inexpensive **low‑fidelity (LF)** simulations with a small number of **high‑fidelity (HF)** simulations to produce accurate, high‑resolution flood hazard maps. The approach operates on **unstructured meshes** and uses **graph neural networks** (MeshGraphNet) for message passing on k‑NN graphs.

---

## ✨ Highlights
- **Hierarchical LF→HF learning**: train an LF GNN on coarse meshes; upsample predictions; train an HF GNN to learn residual corrections.
- **Unstructured-mesh native**: no rasterization required; works directly with HEC‑RAS‑style meshes.
- **Strong accuracy vs. compute**: achieves lower MAE/RRMSE and higher CSI than single‑fidelity GNNs under the same compute budget.  fileciteturn0file0

---

## 🗂️ Repository Layout

```
.
├── README.md
├── constants.py                 # Central experiment configuration (paths, hyperparameters)
├── utils.py                     # Data I/O, graph building, trainers, model loading, LF→HF upsampling
├── train.py                     # End‑to‑end training: LF stage then HF stage
├── inference.py                 # Evaluation, metrics, and plotting (after training)
└── scripts/
    ├── run_hecras_batch.py      # (Optional) HEC‑RAS automation & extraction tooling
    └── synth_events.py          # (Optional) Synthetic hydrograph/hyetograph generation
```

> The code expects three main entry points: `train.py`, `inference.py`, and shared utilities in `utils.py`. Edit `constants.py` to point at your data, checkpoints, and experiment settings.

---

## 🚀 Quick Start

### 1) Environment
We recommend **Python 3.10+** with CUDA‑enabled PyTorch.

```bash
# Create a fresh environment
conda create -n mfgnn python=3.10 -y
conda activate mfgnn

# Core deps (choose versions compatible with your CUDA/PyTorch stack)
pip install torch dgl -f https://data.dgl.ai/wheels/cu$(nvcc --version | awk '/release/ {gsub(/,/, "", $5); print substr($5,3,2)"0"}')/repo.html

# Scientific / utils
pip install numpy scipy scikit-learn pandas h5py matplotlib tqdm

# NVIDIA Modulus (for MeshGraphNet) – pick a version compatible with your CUDA/PyTorch
pip install nvidia-modulus

# Optional: experiment tracking
pip install wandb
```

> **Note:** If `dgl` wheel resolution is tricky, consult the [DGL installation guide](https://www.dgl.ai/pages/start.html) for the exact CUDA/PyTorch matrix, or install the CPU wheel for experimentation.

### 2) Configure paths & hyperparameters
Edit `constants.py` to set data directories, output checkpoints, batch sizes, LF/HF sample counts, learning rates, etc.

Key fields:
- `ckpt_path`, `ckpt_name_lf`, `ckpt_name_hf`
- `data_dir`, `test_dir`, `cache_dir`, `results_dir`
- `input_dim_nodes`, `input_dim_edges`, `output_dim`
- `epochs_LF`, `epochs_HF`, `N_LF`, `N_HF`, `N_Test`

### 3) Prepare data
The code expects tab‑delimited text files for each simulation (per prefix):
- `{prefix}_XY.txt` – coordinates (N×2)  
- `{prefix}_CE.txt` – elevation (N×1)  
- `{prefix}_CA.txt` – cell area (N×1)  
- `{prefix}_CS.txt` – slope (N×1)  
- `{prefix}_A.txt`  – aspect (N×1)  
- `{prefix}_CU.txt` – curvature (N×1)  
- `{prefix}_WD_{ID}.txt` – max water depth time series (T×N)  
- `{prefix}_VX_{ID}.txt`, `{prefix}_VY_{ID}.txt` – velocity components (T×N)  
- `{prefix}_US_InF_{ID}.txt` – inflow hydrograph (T×2: time, value)

`utils.py` computes LF/HF graphs, k‑NN edges, edge features, and normalization stats. Test graphs reuse saved stats to ensure consistent scaling across splits.

> Optional scripts in `scripts/` show how to (a) automate HEC‑RAS runs and HDF5 extraction, and (b) generate synthetic events by scaled cloning of base hydro/ hyetographs.

### 4) Train
```bash
python train.py
```
- Trains **LF** model first; saves `normalization_params_LF.json` and the best LF checkpoint.  
- Builds HF training graphs, **upsamples LF predictions to HF**, appends as a feature, then trains the **HF** model.  
- Periodically saves checkpoints and per‑epoch losses to `ckpt_path`.

### 5) Evaluate & Plot
```bash
python inference.py
```
- Loads best LF/HF models and HF stats.  
- Builds HF test graphs and augments features using LF predictions.  
- Computes **MAE, MSE, RMSE, RRMSE, CSI@0.05/0.3, Continuous CSI**, and per‑sample inference times.  
- Saves images for **Ground Truth**, **Prediction**, **Upsampled LF**, and **Error** maps; also saves **node‑wise error histograms** and `.npy` arrays for reproducibility.

---

## 📦 Data Expectations & Naming

- **Prefixes:** `M80_*` for low‑fidelity (coarse) and `M10_*` for high‑fidelity (fine) simulations (default).  
- **IDs:** Simulation identifiers (e.g., `T001`, `T002`, …) must be consistent across LF/HF datasets for paired upsampling.
- **Normalization:** Training stores min/max stats per feature; test/val strictly reuse them.

> See the paper for node/edge feature definitions, hydrograph descriptors, and CSI metric details.  fileciteturn0file0

---

## 📊 Reported Benefits (from the paper)
- **Iowa River (fluvial)**: MFGNN lowers MAE and boosts CSI vs. a single‑fidelity GNN trained under the same compute budget.  
- **White River (pluvial+fluvial)**: MFGNN yields markedly better MAE/RRMSE/CSI despite far fewer HF runs, demonstrating strong efficiency.  fileciteturn0file0

---

## 🧪 Repro Tips
- Fix seeds in `constants.py` for repeatability.  
- Cache processed graphs via `cache_dir` to avoid repeated preprocessing.  
- Start with smaller `N_LF`, `N_HF`, and a reduced hidden size for quick smoke‑tests.  
- If memory‑bound on very large meshes, reduce batch size or processor depth (GN blocks).

---

## 📚 Citation

If you use this repository, please cite the paper:

```bibtex
@article{Taghizadeh2025MFGNN,
  title   = {Multi-fidelity graph neural networks for efficient and accurate flood hazard mapping},
  author  = {Taghizadeh, Mehdi and Zandsalimi, Zanko and Shafiee-Jood, Majid and Alemazkoor, Negin},
  journal = {Environmental Modelling and Software},
  volume  = {193},
  pages   = {106654},
  year    = {2025},
  doi     = {10.1016/j.envsoft.2025.106654}
}
```

---

## 📄 License
Unless otherwise noted, this work is released under the **Apache-2.0** license (to align with NVIDIA Modulus usage). Adjust if your project uses a different license.

---

## 🙋 Support & Contact
Questions or issues? Please open a GitHub issue. For paper‑related inquiries, contact the corresponding author listed in the manuscript.  fileciteturn0file0
# MFGNN_Flood_Mapping