# Sc_RNA_seq

Single-cell RNA-seq analysis using [ember](https://pypi.org/project/ember-py/) (entropy-based gene specificity metrics) plus an optimized polygon visualization of gene specificity across a categorical partition (developmental age, genotype, etc.).

## `src/` — ember + optimized polygon pipeline

The active pipeline. Run order:

1. **`check_content.py`** — opens a `.h5ad` file in backed mode and prints its `obs`/`var` columns, so you can confirm/fill in `config.py`'s column names before running anything expensive.
2. **`config.py`** — the only file that should need editing per dataset: data path, partition/sample/condition column names, category list, EMBER run parameters (draws, permutation iterations, CPUs), and gene-selection thresholds.
3. **`run_ember.py`** — runs EMBER's `light_ember` (entropy metrics + Psi/Psi_block scores) and `generate_pvals` (empirical p/q-values via permutation testing) on the full dataset, no subsampling.
4. **`polygon_plot.py`** — generates the optimized polygon specificity plot from `run_ember.py`'s output.

**`polygon_specificity.py`** is the actual reusable piece: `plot_polygon_specificity(...)`, written to match `ember_py`'s own function conventions (path-based I/O, `partition_label`/`save_dir` arguments). It implements an ILR &rarr; PCA &rarr; Procrustes-aligned polygon layout.

**`polygon_plot_tutorial.ipynb`** — an already-executed, runnable walkthrough of the whole workflow, using real ember output already in this repo (`output_results/`), so it can be read or re-run without needing your own dataset first.

### Requirements

- Python **3.9+**
- `pip install ember-py scanpy anndata pandas numpy matplotlib scikit-learn statsmodels tqdm seaborn h5py`

### Data

The `.h5ad` input file is not tracked in this repository. Point `config.py`'s `DATA_FILE` at wherever it lives on your system/server.

## Other files

- **`ILR.py`** — an earlier draft of the ILR/PCA/Procrustes polygon method, superseded by `src/polygon_specificity.py`. Kept for history; not maintained.
