"""
Run this on h5ad file.

Opens the file in backed mode (does not load the full expression matrix into
memory) and prints exactly what's needed to fill in the TODOs in config.py:
  - obs columns and, for likely strain/sample/condition columns, their values
  - var columns (to find the gene-symbol column, if any)
  - shape, to sanity check nothing was silently truncated on read
"""
import scanpy as sc
import config as cfg


def main():
    print(f"Opening {cfg.DATA_FILE} in backed mode")
    adata = sc.read_h5ad(cfg.DATA_FILE, backed='r')

    print(f"\nShape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    print("\nobs columns:")
    for col in adata.obs.columns:
        print(f"  {col!r}  (dtype={adata.obs[col].dtype})")

    print("\nvar columns:")
    for col in adata.var.columns:
        print(f"  {col!r}")

    print(f"\nvar index preview (first 5):")
    print(list(adata.var.index[:5]))

    # Best-effort guess at which obs columns look like strain / sample / condition,
    # so you can cross-check against config.py's CATEGORY_COL / SAMPLE_ID_COL / CONDITION_COL.
    print("\ncandidate columns (low-cardinality, likely categorical)")
    for col in adata.obs.columns:
        try:
            n_unique = adata.obs[col].nunique()
        except TypeError:
            continue
        if 1 < n_unique <= 30:
            values = sorted(map(str, adata.obs[col].unique()))
            print(f"  {col!r}: {n_unique} unique values -> {values}")


if __name__ == '__main__':
    main()
