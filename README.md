# TCGA/CPTAC Kidney Cancer Single-Cell RNA Sequencing Analysis

This project performs single-cell RNA sequencing (scRNA-seq) analysis on kidney cancer datasets from [TCGA](https://portal.gdc.cancer.gov/) and [CPTAC-3](https://proteomics.cancer.gov/programs/cptac). The pipeline leverages **scvi-tools** for probabilistic batch-corrected integration and **Scanpy** for downstream clustering, differential expression, and cell type annotation.

## Overview

The analysis identifies and annotates cell populations in kidney tumors, including:

| Cell Type | Key Markers |
|---|---|
| Tumor / Neoplastic (ccRCC) | CA9, HIF1A, VHL, VEGFA |
| Proximal Tubule | CUBN, LRP2, SLC34A1 |
| Loop of Henle / TAL | UMOD, SLC12A1 |
| Collecting Duct | AQP2, ATP6V1B1 |
| Podocytes | NPHS1, NPHS2, PODXL |
| Endothelial | PECAM1, VWF, CDH5, FLT1 |
| Fibroblasts / Mesangial | DCN, ACTA2, PDGFRB |
| T cells | CD3D, CD3E, CD8A, CD4, IL7R |
| Macrophages / Monocytes | CD68, AIF1, CD163, LYZ |
| B cells | CD79A, MS4A1 |

## Project Structure

```
├── TCGA_Kidney_Preprocessing.ipynb       # Data download, extraction, and preprocessing
├── kidney_model_training.py              # Standalone scVI model training script
├── kidney_model_training_archive.py      # Archived previous training configuration
├── scVI_Kidney_analysis.ipynb            # Full analysis: QC → scVI → clustering → annotation
├── requirements.txt                      # Python dependencies
└── README.md
```

### File Descriptions

- **`TCGA_Kidney_Preprocessing.ipynb`** — Downloads and preprocesses raw scRNA-seq data from TCGA/CPTAC-3. Extracts count matrices, builds a unified AnnData object, and performs SCTransform normalization.

- **`kidney_model_training.py`** — A standalone Python script for training the scVI variational autoencoder model. Designed for headless execution (e.g., scheduled jobs or remote servers).

- **`scVI_Kidney_analysis.ipynb`** — The main analysis notebook. Performs quality control, highly variable gene (HVG) selection, scVI model training, batch-corrected UMAP visualization, Leiden clustering at multiple resolutions, Bayesian differential expression, and cell type annotation using canonical kidney and RCC marker genes.

## Pipeline

```
Raw Counts (.h5ad)
    │
    ▼
Quality Control & Filtering (MAD-based outlier removal, MT% < 20%)
    │
    ▼
HVG Selection (Seurat v3, top 4000 genes, batch-aware)
    │
    ▼
scVI Model Training (15 latent dims, 2 layers, negative binomial likelihood)
    │
    ▼
Latent Space → Neighbors → UMAP
    │
    ▼
Leiden Clustering (multiple resolutions: 0.3, 0.5, 0.8, 1.0)
    │
    ▼
Bayesian Differential Expression (scVI)
    │
    ▼
Cell Type Annotation (canonical marker gene scoring)
    │
    ▼
Annotated AnnData (.h5ad)
```

## Setup

### 1. Create a Conda Environment

```bash
conda create -n tcga_kidney python=3.11 -y
conda activate tcga_kidney
pip install -r requirements.txt
```

### 2. Set Environment Variables

The preprocessing notebook reads the raw data tarball path from an environment variable to avoid hardcoding local file paths:

```bash
export KIDNEY_TARBALL_PATH="/path/to/your/gdc_download.tar.gz"
```

Add this line to your `~/.zshrc` (macOS) or `~/.bashrc` (Linux) to persist it across sessions.

### 3. Run the Analysis

**Option A — Interactive (recommended):**
Open and run the notebooks sequentially in Jupyter:
1. `TCGA_Kidney_Preprocessing.ipynb`
2. `scVI_Kidney_analysis.ipynb`

**Option B — Headless model training:**
```bash
python kidney_model_training.py
```

## Hardware

This pipeline was developed and tested on:
- **Mac Studio** with **Apple M4 Max**
- Model training uses **MPS** (Metal Performance Shaders) acceleration
- To run on a CUDA GPU, change `accelerator="mps"` to `accelerator="cuda"` in the training cells/script

## Data

Raw data is sourced from the [GDC Data Portal](https://portal.gdc.cancer.gov/). Due to file size, the `.h5ad` data files and trained model weights are **not** included in this repository.

## License

This project is for academic and research purposes.
