
"""
Runs ember on h5ad file with proper partition label and no subsampling.
Run check_content.py and fix config.py's TODOs before running this.
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import subprocess

import pandas as pd
import scanpy as sc

try:
    import ember_py
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ember-py'], check=True)

from ember_py.light_ember import light_ember
from ember_py.plots import plot_sample_counts

import config as cfg

GENE_SYMBOL_CANDIDATES = ['gene_symbols', 'feature_name', 'gene_symbol', 'symbol']


def write_gene_symbol_lookup(data_file, output_dir):
    """
    Save a small (genes-only) lookup table mapping whatever the .h5ad's
    var index is (often Ensembl IDs) to a readable gene symbol column, without
    touching the expression matrix. Cheap even on a huge file since var is
    just a per-gene table.
    """
    adata = sc.read_h5ad(data_file, backed='r')
    symbol_col = next((c for c in GENE_SYMBOL_CANDIDATES if c in adata.var.columns), None)

    lookup = pd.DataFrame(index=adata.var.index)
    lookup.index.name = 'gene_name'
    if symbol_col is not None:
        lookup['gene_symbol'] = adata.var[symbol_col].values
        print(f"Gene symbol column found: '{symbol_col}'.")
    else:
        lookup['gene_symbol'] = lookup.index
        print("No gene-symbol column found in .var; downstream scripts will "
              "use the var index as-is (check_content.py var columns list).")

    out_path = os.path.join(output_dir, 'gene_symbol_lookup.csv')
    lookup.to_csv(out_path)
    print(f"Saved gene symbol lookup to {out_path}")


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(cfg.DATA_FILE):
        raise FileNotFoundError(
            f"{cfg.DATA_FILE} not found. Update config.DATA_FILE or place the file there."
        )

    print("Generating sample count sanity-check plot...")
    try:
        plot_sample_counts(
            h5ad_dir=cfg.DATA_FILE,
            save_dir=cfg.OUTPUT_DIR,
            sample_id_col=cfg.SAMPLE_ID_COL,
            category_col=cfg.CATEGORY_COL,
            condition_col=cfg.CONDITION_COL,
        )
    except Exception as e:
        print(f"plot_sample_counts failed (check config.py column names): {e}")

    print("Saving gene symbol lookup")
    write_gene_symbol_lookup(cfg.DATA_FILE, cfg.OUTPUT_DIR)

    print("Running light_ember (full dataset, no cell subsampling)")
    light_ember(
        h5ad_dir=cfg.DATA_FILE,
        partition_label=cfg.PARTITION_LABEL,
        save_dir=cfg.OUTPUT_DIR,
        sampling=True,
        sample_id_col=cfg.SAMPLE_ID_COL,
        category_col=cfg.CATEGORY_COL,
        condition_col=cfg.CONDITION_COL,
        num_draws=cfg.NUM_DRAWS,
        partition_pvals=True,
        block_pvals=False,
        n_pval_iterations=cfg.N_PVAL_ITERATIONS,
        n_cpus=cfg.N_CPUS,
    )

    print("Done.")
    print(f"Psi_block matrix: {cfg.OUTPUT_DIR}/Psi_block_df/mean_Psi_block_df_{cfg.PARTITION_LABEL}.csv")
    print(f"P-values:         {cfg.OUTPUT_DIR}/pvals_entropy_metrics_{cfg.PARTITION_LABEL}.csv")
    print("Next: run polygon_plot.py")


if __name__ == '__main__':
    main()
