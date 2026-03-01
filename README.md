# TCGA Kidney Cancer scRNA-seq Analysis

Single-cell RNA sequencing (scRNA-seq) analysis of **Kidney Cancer** samples — a subset from the [CPTAC-3 (Clinical Proteomic Tumor Analysis Consortium)](https://portal.gdc.cancer.gov/projects/CPTAC-3) project on the GDC Data Portal — using probabilistic deep learning with **scvi-tools**.

This project performs batch-corrected integration of kidney cancer patient samples, Bayesian differential expression, and unsupervised cell type annotation — providing a reproducible framework for dissecting the kidney tumor microenvironment at single-cell resolution.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Analysis Pipeline](#analysis-pipeline)
- [Getting Started](#getting-started)
- [Hardware Acceleration](#hardware-acceleration)
- [Requirements](#requirements)
- [Results](#results)
- [License](#license)

---

## Overview

Kidney cancer encompasses several subtypes, with clear cell renal cell carcinoma (ccRCC) being the most prevalent. This project leverages scRNA-seq data from the CPTAC-3 project on GDC to:

1. **Construct an h5ad dataset** from raw GDC downloads (11 cases, 19 samples)
2. **Integrate across batches** using a Variational Autoencoder ([scVI](https://docs.scvi-tools.org/en/stable/)) to remove technical confounders while preserving biological variation
3. **Identify cell populations** via Leiden clustering on the batch-corrected latent space
4. **Perform differential expression** using scVI's Bayesian framework for calibrated uncertainty estimates
5. **Annotate cell types** with canonical kidney and RCC marker genes

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

---

## Project Structure

```
TCGA_Kidney_scRNA-seq/
│
├── TCGA_Kidney_Preprocessing.ipynb       # Data ingestion — builds h5ad from GDC downloads
├── scVI_Kidney_analysis.ipynb            # Full analysis — QC, scVI integration, clustering, DE, annotation
├── kidney_model_training.py              # Standalone scVI training script (MPS)
├── kidney_model_training_archive.py      # Archived previous training configuration
│
├── gdc_sample_sheet.2026-03-01.tsv       # GDC sample sheet for reproducing the 11-case dataset
├── scvi_model_kidney/                    # Saved scVI model checkpoint
├── requirements.txt                      # Python dependencies
│
└── README.md
```

---

## Data

**Source:** A Kidney Cancer subset from the [CPTAC-3 project](https://portal.gdc.cancer.gov/projects/CPTAC-3) on the [GDC Data Portal](https://portal.gdc.cancer.gov/)

- **11 patient cases** (19 sample files) of scRNA-seq data downloaded from GDC (CPTAC-3)
- Raw data is extracted and assembled into an [AnnData](https://anndata.readthedocs.io/) `.h5ad` object in `TCGA_Kidney_Preprocessing.ipynb`
- The assembled dataset (`tcga_cptac_kidney_data.h5ad`) contains:
  - **Raw integer counts** in `adata.layers["counts"]` (required by scVI)
  - **SCT-normalized expression** in `adata.X`
  - **Batch labels** per patient in `adata.obs["batch"]`

> **Note:** The `.h5ad` data files and raw GDC downloads are excluded from version control due to their size. Use the preprocessing notebook and steps below to reproduce the dataset locally.

### Reproducing the Dataset

The included [`gdc_sample_sheet.2026-03-01.tsv`](gdc_sample_sheet.2026-03-01.tsv) contains the exact 11 cases used in this analysis. All samples are **primary kidney tumors** stored as Seurat `.loom` files.

<details>
<summary><b>11 CPTAC-3 Case IDs (19 samples)</b> (click to expand)</summary>

| File ID | Case ID | Sample ID | Tissue Type |
|--------------------------------------|------------|--------------|-------------|
| 61ca2fed-0a04-4987-a7fa-a1f7151f3ee1 | C3N-01270 | C3N-01270-02 | Tumor |
| 069690ba-d3c3-441f-97d4-a17ad446e4a0 | C3N-00148 | C3N-00148-03 | Tumor |
| 7f1eb380-9931-4f13-8756-e9dd3734e7ca | C3N-00148 | C3N-00148-01 | Tumor |
| 276fb5ff-5a94-452a-a72f-099a39e64766 | C3N-00148 | C3N-00148-04 | Tumor |
| ab005d54-0e8b-4387-abb5-3b4fae1713ad | C3L-00606 | C3L-00606-02 | Tumor |
| b87a690b-6a0c-493b-9b60-253fc849ed03 | C3L-00606 | C3L-00606-03 | Tumor |
| ba7a2eef-b3a4-4144-aa5c-8e1c7c2987c4 | C3L-01953 | C3L-01953-01 | Tumor |
| 5c1cc7db-43bb-4b22-aa06-f46540a071eb | C3N-00148 | C3N-00148-02 | Tumor |
| 0fa50788-e27c-4f9c-a1a8-be2e8686c000 | C3L-00606 | C3L-00606-01 | Tumor |
| 2ae3a4c7-a8ff-4e54-96b2-02e2b0424e0b | C3N-01904 | C3N-01904-02 | Tumor |
| a02e0dbe-3e7f-4b72-8e05-c60ddc4cec52 | C3L-02858 | C3L-02858-01 | Tumor |
| c5993768-696a-4cfb-b1a7-2cb5329b2da2 | C3L-01287 | C3L-01287-03 | Tumor |
| 18022782-fedc-4f43-9900-e636649b1f09 | C3L-00359 | C3L-00359-01 | Tumor |
| 1f0f8468-2d4a-4599-bc24-eb3af2888cbb | C3N-00149 | C3N-00149-03 | Tumor |
| bd195b88-bf31-43b2-b145-28debb9a0d4b | C3N-00149 | C3N-00149-02 | Tumor |
| bd391d25-da52-4da4-9c45-e685ae491ec1 | C3N-00439 | C3N-00439-02 | Tumor |
| d8011f6a-844f-45a4-9cc7-7ac8ccf1523d | C3L-01287 | C3L-01287-01 | Tumor |
| 92bfb2b2-c4e7-4a0b-84c2-dc8467342b19 | C3N-00149 | C3N-00149-04 | Tumor |
| c8b631b1-9f05-4172-827b-8ab7d13cb40e | C3N-01175 | C3N-01175-01 | Tumor |

</details>

**To download the data from GDC:**

1. Go to the [GDC Data Portal](https://portal.gdc.cancer.gov/) and navigate to the **CPTAC-3** project
2. Filter for **Data Category:** Transcriptome Profiling → **Data Type:** Single Cell Analysis
3. Select the 11 cases listed above (or import the sample sheet directly)
4. Download the files using the GDC Data Transfer Tool or the portal's cart
5. Set the tarball path as an environment variable:

```bash
export KIDNEY_TARBALL_PATH="/path/to/your/gdc_download.tar.gz"
```

6. Run `TCGA_Kidney_Preprocessing.ipynb` to assemble `tcga_cptac_kidney_data.h5ad`

Add the export line to your `~/.zshrc` (macOS) or `~/.bashrc` (Linux) to persist it across sessions.

---

## Analysis Pipeline

### Notebook 1 — `TCGA_Kidney_Preprocessing.ipynb` (Data Preparation)

| Step | Description |
|------|-------------|
| 1 | Define paths and inspect the GDC tarball |
| 2 | Extract raw scRNA-seq files |
| 3 | Auto-detect file types and load per-sample data |
| 4 | Assemble multi-sample AnnData and save as `tcga_cptac_kidney_data.h5ad` |

### Notebook 2 — `scVI_Kidney_analysis.ipynb` (Integration & Annotation)

| Step | Description |
|------|-------------|
| 1 | Load data and setup environment |
| 2 | Quality control — mitochondrial/ribosomal gene detection, MAD-based outlier removal, MT% < 20% filtering |
| 3 | Highly variable gene (HVG) selection (`seurat_v3`, 4 000 genes, batch-aware) |
| 4 | scVI model setup and training (15-dim latent space, negative binomial likelihood) |
| 5 | Latent space extraction, neighbor graph, and UMAP visualization |
| 6 | Leiden clustering at multiple resolutions (0.3, 0.5, 0.8, 1.0) |
| 7 | Bayesian differential expression with scVI |
| 8 | Cell type annotation using canonical kidney and RCC marker genes |
| 9 | Save annotated AnnData and trained model |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- [Conda](https://docs.conda.io/) or [pip](https://pip.pypa.io/) for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/jjevans25/TCGA_Kidney_scRNA-seq.git
cd TCGA_Kidney_scRNA-seq

# Create a Conda environment
conda create -n tcga_kidney python=3.11 -y
conda activate tcga_kidney

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the Analysis

1. **Prepare the data** — Run `TCGA_Kidney_Preprocessing.ipynb` to build `tcga_cptac_kidney_data.h5ad` from raw GDC files
2. **Train the scVI model** — Use the standalone training script, which covers QC, HVG selection, and model training/saving:

```bash
python kidney_model_training.py
```

3. **Run downstream analysis** — Open `scVI_Kidney_analysis.ipynb` for the full pipeline, including latent space visualization, Leiden clustering, differential expression, and cell type annotation

---

## Hardware Acceleration

The standalone training script (`kidney_model_training.py`) is configured for **MPS** (Metal Performance Shaders) on Apple Silicon. To use a different accelerator, update the `accelerator` parameter in the `model.train()` call (e.g., `"cuda"` for NVIDIA GPU, `"cpu"` for CPU-only).

Training parameters:
- **Max epochs:** 500 (with early stopping, patience = 20)
- **Latent dimensions:** 15
- **Network depth:** 2 layers
- **Likelihood:** Negative binomial
- **Batch size:** 256

---

## Requirements

Core libraries and their roles:

| Library | Purpose |
|---------|---------|
| [scanpy](https://scanpy.readthedocs.io/) | Preprocessing, clustering, visualization |
| [scvi-tools](https://docs.scvi-tools.org/) | Probabilistic modeling & batch integration |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [leidenalg](https://leidenalg.readthedocs.io/) | Community detection (clustering) |
| [requests](https://docs.python-requests.org/) | MyGene.info API for Ensembl ID → gene symbol mapping |

See [`requirements.txt`](requirements.txt) for the full dependency list.

---

## Results

The final annotated dataset contains:

- **Batch-corrected UMAP embeddings** revealing biologically meaningful structure across patient samples
- **Leiden cluster assignments** at multiple resolutions
- **Bayesian differential expression** results per cluster
- **Cell type annotations** mapped via canonical markers for kidney cancer–relevant populations (e.g., ccRCC tumor cells, proximal tubule, immune cells, endothelial cells)

---

## License

This project is for academic and research purposes. CPTAC-3 data is publicly available through the GDC and subject to the [GDC Data Use Agreement](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).
