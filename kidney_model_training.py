# For scheduled job training

import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# Settings
sc.settings.verbosity = 2
sc.settings.set_figure_params(
    dpi=100, frameon=False, figsize=(6, 5), facecolor="white"
)
sc.settings.n_jobs = -1
sc.settings.num_workers = 13
scvi.settings.num_workers = 13
scvi.settings.seed = 42

# Paths specific to Kidney dataset
DATA_PATH = "tcga_cptac_kidney_data.h5ad"
MODEL_DIR = "scvi_model_kidney"

print(f"scvi-tools version: {scvi.__version__}")
print(f"scanpy version: {sc.__version__}")

# Load the data
adata = sc.read_h5ad(DATA_PATH)
print(adata)
print(f"\nLayers: {list(adata.layers.keys())}")
print(f"obs columns: {list(adata.obs.columns)}")

# Verify we have raw counts in the 'counts' layer
# scVI requires integer counts as input
counts_sample = adata.layers["counts"][:5, :5]
if hasattr(counts_sample, "toarray"):
    counts_sample = counts_sample.toarray()
print("Sample counts (should be integers):")
print(counts_sample)
print(f"\nAll integer values: {np.allclose(counts_sample, counts_sample.astype(int))}")

# Ensure obs_names are unique
adata.obs_names_make_unique()
adata.var_names_make_unique()

# Ensure batch is categorical
adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
print(f"Number of batches: {adata.obs['batch'].nunique()}")
print(f"Cells per batch:\n{adata.obs['batch'].value_counts()}")

# Use raw counts for QC
# Store SCT-normalized X for reference, then swap in raw counts
adata.layers["sct_normalized"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

# Identify mitochondrial and ribosomal genes by Ensembl ID patterns
try:
    print("Querying MyGene.info API for gene symbols...")
    # Strip version numbers from Ensembl IDs
    adata.var["ensembl_id"] = adata.var_names
    adata.var["ensembl_id_no_version"] = adata.var_names.str.split(".").str[0]
    
    ensembl_list = adata.var["ensembl_id_no_version"].unique().tolist()
    
    # MyGene.info recommends batching POST requests (max 1000 per request)
    chunk_size = 1000
    results = []
    
    for i in range(0, len(ensembl_list), chunk_size):
        chunk = ensembl_list[i:i + chunk_size]
        res = requests.post(
            "https://mygene.info/v3/query",
            data={"q": ",".join(chunk), "scopes": "ensembl.gene", "fields": "symbol,genomic_pos.chr", "species": "human"}
        )
        if res.status_code == 200:
            results.extend(res.json())
        else:
            print(f"Warning: API request failed with status {res.status_code}")
            
    if results:
        # Handle multiple hits by taking the first one
        gene_info = pd.DataFrame(results).drop_duplicates(subset="query")
        
        # Extract chromosome properly since genomic_pos can be a list or dict
        def get_chr(row):
            pos = row.get('genomic_pos')
            if isinstance(pos, list) and len(pos) > 0:
                return pos[0].get('chr')
            elif isinstance(pos, dict):
                return pos.get('chr')
            return None
            
        gene_info['chromosome'] = gene_info.apply(get_chr, axis=1)
        
        # Merge gene info
        adata.var = adata.var.merge(
            gene_info.rename(columns={
                "query": "ensembl_id_no_version",
                "symbol": "gene_symbol"
            })[["ensembl_id_no_version", "gene_symbol", "chromosome"]],
            on="ensembl_id_no_version",
            how="left"
        )
        adata.var.index = adata.var["ensembl_id"]
        
        # Flag mitochondrial genes
        adata.var["mt"] = adata.var["chromosome"].fillna("").astype(str).str.upper() == "MT"
        adata.var["ribo"] = adata.var["gene_symbol"].fillna("").str.upper().str.match(r"^RP[SL]")
        print(f"Mapped {adata.var['gene_symbol'].notna().sum()} / {adata.n_vars} genes to symbols")
        print(f"MT genes: {adata.var['mt'].sum()}")
        print(f"Ribosomal genes: {adata.var['ribo'].sum()}")
    else:
        raise ValueError("No results returned from API")
    
except Exception as e:
    print(f"API Mapping failed: {e}. Using heuristic MT gene detection.")
    # Fallback: known human MT gene Ensembl IDs
    mt_ensembl = [
        "ENSG00000198888", "ENSG00000198763", "ENSG00000198804",
        "ENSG00000198712", "ENSG00000228253", "ENSG00000198899",
        "ENSG00000198938", "ENSG00000198840", "ENSG00000212907",
        "ENSG00000198886", "ENSG00000198786", "ENSG00000198695",
        "ENSG00000198727"
    ]
    if "ensembl_id_no_version" not in adata.var:
        adata.var["ensembl_id_no_version"] = adata.var_names.str.split(".").str[0]
    adata.var["mt"] = adata.var["ensembl_id_no_version"].isin(mt_ensembl)
    adata.var["ribo"] = False  # Can't reliably detect without symbols
    print(f"MT genes found: {adata.var['mt'].sum()}")

# Select HVGs using seurat_v3 method (works on raw counts)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="batch",
    subset=False,
    layer="counts"
)

print(f"Highly variable genes: {adata.var['highly_variable'].sum()}")

# Subset to highly variable genes for scVI
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
print(f"HVG subset shape: {adata_hvg.shape}")

# Setup scVI model
scvi.model.SCVI.setup_anndata(
    adata_hvg,
    layer="counts",
    batch_key="batch",
)

# Initialize the model
model = scvi.model.SCVI(
    adata_hvg,
    n_latent=15,
    n_layers=2,
    gene_likelihood="nb",  # Negative binomial for count data
)

print(model)

# Train the model
# Using MPS (Metal Performance Shaders) acceleration for Apple Silicon / M4 Max
model.train(
    max_epochs=500,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_monitor="elbo_validation",
    accelerator="mps",
    devices=1,
    batch_size=256,
)

# Save the trained model
model.save(MODEL_DIR, overwrite=True)
print(f"Model saved to: {MODEL_DIR}")