"""
Generate the optimized polygon plot for desired ember output.

All the actual logic (ILR -> PCA -> Procrustes polygon alignment) lives in polygon_specificity.py, written to match ember_py's own function conventions (path-based I/O, partition_label + save_dir, etc.) so it can be dropped into ember_py/plots.py directly.
This script is just "run it for our config".
"""
import os
import pandas as pd

from polygon_specificity import plot_polygon_specificity
import config as cfg

LOOKUP_PATH = os.path.join(cfg.OUTPUT_DIR, 'gene_symbol_lookup.csv')


def main():
    gene_symbol_lookup = None
    if os.path.exists(LOOKUP_PATH):
        gene_symbol_lookup = pd.read_csv(LOOKUP_PATH, index_col=0)['gene_symbol']

    plot_polygon_specificity(
        partition_label=cfg.PARTITION_LABEL,
        psi_block_dir=os.path.join(cfg.OUTPUT_DIR, 'Psi_block_df'),
        pvals_dir=os.path.join(cfg.OUTPUT_DIR, f'pvals_entropy_metrics_{cfg.PARTITION_LABEL}.csv'),
        save_dir=cfg.OUTPUT_DIR,
        categories=cfg.CATEGORIES,
        category_colors=cfg.CATEGORY_COLORS,
        psi_thresh=cfg.PSI_THRESH,
        p_thresh=cfg.P_THRESH,
        q_thresh=cfg.Q_THRESH,
        n_single=cfg.N_SINGLE,
        n_per_pair=cfg.N_PER_PAIR,
        min_psi=cfg.MIN_PSI,
        gene_symbol_lookup=gene_symbol_lookup,
    )


if __name__ == '__main__':
    main()
