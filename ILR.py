import matplotlib
matplotlib.use('Agg')
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import permutations

PSI_PATH    = os.path.join('output_results', 'Psi_block_df', 'mean_Psi_block_df_Age.csv')
OUTPUT_DIR  = 'output_results'
N_TOP_GENES = 500   # genes used for ILR/PCA fitting
N_SINGLE    = 10    # top genes per age (single-age specific)
N_MULTI     = 15    # pairwise/triple specific genes
DEV_ORDER   = ['E16.5', 'P0', 'W3', 'W12', 'W52', 'W92']

# One colour per age
AGE_COLORS = {
    'E16.5': '#E41A1C',
    'P0':    '#377EB8',
    'W3':    '#4DAF4A',
    'W12':   '#984EA3',
    'W52':   '#FF7F00',
    'W92':   '#A65628',
}
MULTI_COLOR  = '#555555'   # dark grey for pairwise/triple genes
MULTI_MARKER = '^'         # triangle


#ILR transform (this is AI generated, should double check)
def ilr_transform(X):
    """Sequential-binary-partition ILR transform.  X: (n, K) -> (n, K-1)."""
    K = X.shape[1]
    X = np.clip(X, 1e-9, None)
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    ilr = np.zeros((X.shape[0], K - 1))
    for i in range(1, K):
        ilr[:, i - 1] = np.sqrt(i / (i + 1)) * (log_X[:, :i].mean(axis=1) - log_X[:, i])
    return ilr


# Procrustes
def procrustes_rotation(A, B):
    """Align A to B.  Returns A_aligned, R (2x2), scale s, residual MSE."""
    A_c = A - A.mean(0)
    B_c = B - B.mean(0)
    U, S, Vt = np.linalg.svd(B_c.T @ A_c)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    s = float((B_c * (A_c @ R.T)).sum() / (A_c ** 2).sum())
    A_aligned = s * A_c @ R.T + B.mean(0)
    residual  = float(np.mean((A_aligned - B) ** 2))
    return A_aligned, R, s, residual


# find best vertex assignment by trying all permutations
def best_vertex_assignment(pca_coords, single_genes, cats, K):
    """
    Try all K! permutations.  For each, Procrustes-align the K polygon vertex
    positions to the K per-age PCA centroids (centroid of single-age genes).
    This ensures age-specific genes land near their vertex, not a global fit.
    """
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    verts  = np.column_stack([np.cos(angles), np.sin(angles)])

    # One PCA centroid per age
    pca_centroids = np.array([pca_coords[single_genes[cat]].mean(0)
                               for cat in cats])

    best_perm, best_res = list(range(K)), np.inf
    for perm in permutations(range(K)):
        # perm[i] = which category index sits at vertex i
        # => category j sits at vertex inv_perm[j]
        inv_perm = [0] * K
        for vi, ci in enumerate(perm):
            inv_perm[ci] = vi
        poly_centroids = np.array([verts[inv_perm[j]] for j in range(K)])
        _, _, _, res = procrustes_rotation(poly_centroids, pca_centroids)
        if res < best_res:
            best_res, best_perm = res, list(perm)
    return best_perm, best_res


# Select top genes for single-age and multi-age specific genes
def select_highlight_genes(W_norm, gene_names, cats, n_single=10, n_multi=15):
    """
    Group 1 — single-age specific: top n_single genes per age by Psi score.
              No gene appears in more than one age group.
    Group 2 — multi-age specific:  genes not in group 1 with the highest
              sum of their top-2 Psi scores (shared between 2-3 ages).

    Returns
    single : dict  {age_label: [gene_index, ...]}
    multi  : list  [gene_index, ...]
    """
    used = set()

    single = {}
    for j, cat in enumerate(cats):
        order = np.argsort(W_norm[:, j])[::-1]
        selected = []
        for idx in order:
            if idx not in used and len(selected) < n_single:
                selected.append(int(idx))
                used.add(int(idx))
        single[cat] = selected


    candidates = []
    for i in range(len(gene_names)):
        if i in used:
            continue
        sorted_w = np.sort(W_norm[i])[::-1]
        candidates.append((i, float(sorted_w[0] + sorted_w[1])))
    candidates.sort(key=lambda x: x[1], reverse=True)
    multi = [idx for idx, _ in candidates[:n_multi]]

    return single, multi


# Plotting helper function
def draw_panel(ax, coords, gene_names, cats, single_genes, multi_genes,
               age_colors, panel='pca', verts_aligned=None, cat_at_vertex=None,
               W_norm=None, var_exp=None):
    """Shared drawing logic for both panels."""

    # all 500 genes in grey
    ax.scatter(coords[:, 0], coords[:, 1],
               c='lightgray', s=10, alpha=0.4, zorder=1, rasterized=True)

    # Single-age specific genes: circles & coloured by age
        idxs = single_genes[cat]
        col  = age_colors.get(cat, '#333333')
        xs, ys = coords[idxs, 0], coords[idxs, 1]
        ax.scatter(xs, ys, color=col, s=55, marker='o',
                   edgecolor='k', lw=0.4, zorder=3, label=f'{cat}')
        # Label the top 3 per age to keep the plot readable
        for idx in idxs[:3]:
            ax.annotate(gene_names[idx], (coords[idx, 0], coords[idx, 1]),
                        fontsize=6, color=col,
                        xytext=(4, 4), textcoords='offset points')

    # Multi-age specific genes: triangles & dark grey
    xs_m = coords[multi_genes, 0]
    ys_m = coords[multi_genes, 1]
    ax.scatter(xs_m, ys_m, color=MULTI_COLOR, s=65, marker=MULTI_MARKER,
               edgecolor='k', lw=0.4, zorder=4, label='multi-age')
    for idx in multi_genes[:5]:
        ax.annotate(gene_names[idx], (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, color=MULTI_COLOR,
                    xytext=(4, 4), textcoords='offset points')

    if panel == 'pca':
        # Biplot arrows: Pearson r of each age's Psi with PC1/PC2
        span = float(np.ptp(coords, axis=0).max()) * 0.38
        for j, cat in enumerate(cats):
            r1 = float(np.corrcoef(W_norm[:, j], coords[:, 0])[0, 1])
            r2 = float(np.corrcoef(W_norm[:, j], coords[:, 1])[0, 1])
            ax.annotate('', xy=(r1 * span, r2 * span), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5))
            ax.text(r1 * span * 1.18, r2 * span * 1.18, cat,
                    fontsize=8, color='steelblue', ha='center', va='center')
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var. expl.)')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var. expl.)')
        ax.set_title('(a) PCA of ILR(Psi) — accurate geometry', fontsize=11)

    else:  # polygon panel
        # Polygon outline
        outline = np.vstack([verts_aligned, verts_aligned[0]])
        ax.plot(outline[:, 0], outline[:, 1], 'k-', lw=1.8, alpha=0.5, zorder=2)
        center = verts_aligned.mean(0)
        for i, (x, y) in enumerate(verts_aligned):
            cat_v = cat_at_vertex[i]
            col_v = age_colors.get(cat_v, 'k')
            ax.scatter(x, y, color=col_v, s=80, edgecolor='k', lw=0.6, zorder=5)
            lx, ly = center + (np.array([x, y]) - center) * 1.25
            ax.text(lx, ly, cat_v, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=col_v)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('(b) Polygon (ILR-aligned, optimal vertex order)', fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    seen, h_out, l_out = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h_out.append(h); l_out.append(l)
    ax.legend(h_out, l_out, fontsize=7, bbox_to_anchor=(1.02, 1),
              loc='upper left', title='Gene group', title_fontsize=8, framealpha=0.8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_psi = pd.read_csv(PSI_PATH, index_col=0)

    cats  = [c for c in DEV_ORDER if c in df_psi.columns]
    cats += [c for c in df_psi.columns if c not in cats]
    K     = len(cats)
    df_psi = df_psi[cats].dropna()

    specificity = df_psi.max(axis=1) - df_psi.min(axis=1)
    df_top      = df_psi.loc[specificity.nlargest(N_TOP_GENES).index]

    W          = df_top.values.astype(float)
    W_norm     = W / (W.sum(axis=1, keepdims=True) + 1e-9)
    gene_names = df_top.index.tolist()

    # ILR + PCA
    print(f"ILR: {len(gene_names)} genes x {K} categories -> {K-1} coords")
    ilr        = ilr_transform(W_norm)
    pca        = PCA(n_components=2)
    pca_coords = pca.fit_transform(ilr)
    var_exp    = pca.explained_variance_ratio_ * 100
    print(f"PCA: PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%")

    # Gene Selection
    single_genes, multi_genes = select_highlight_genes(
        W_norm, gene_names, cats, N_SINGLE, N_MULTI)

    n_single_total = sum(len(v) for v in single_genes.values())
    print(f"Highlighted: {n_single_total} single-age genes  +  {len(multi_genes)} multi-age genes")

    # Optimal vertex assignment
    print(f"Searching {math.factorial(K)} vertex permutations...")
    best_perm, res = best_vertex_assignment(pca_coords, single_genes, cats, K)
    cat_at_vertex  = [cats[best_perm[i]] for i in range(K)]
    print(f"Best vertex order: {cat_at_vertex}  (residual={res:.5f})")

    # Polygon coords + centroid-based Procrustes align
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    verts  = np.column_stack([np.cos(angles), np.sin(angles)])

    # inv_perm[j] = which vertex index category j maps to
    inv_perm = [0] * K
    for vi, ci in enumerate(best_perm):
        inv_perm[ci] = vi

    # Fit Procrustes: polygon vertex positions -> per-age PCA centroids
    pca_centroids  = np.array([pca_coords[single_genes[cat]].mean(0) for cat in cats])
    poly_centroids = np.array([verts[inv_perm[j]] for j in range(K)])
    _, R, s, _ = procrustes_rotation(poly_centroids, pca_centroids)
    cent_mean  = poly_centroids.mean(0)
    pca_mean   = pca_centroids.mean(0)

    # Apply same transform to all gene polygon positions and vertices
    poly_coords   = W_norm[:, best_perm] @ verts
    poly_aligned  = s * (poly_coords  - cent_mean) @ R.T + pca_mean
    verts_aligned = s * (verts        - cent_mean) @ R.T + pca_mean

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    draw_panel(axes[0], pca_coords, gene_names, cats,
               single_genes, multi_genes, AGE_COLORS,
               panel='pca', W_norm=W_norm, var_exp=var_exp)

    draw_panel(axes[1], poly_aligned, gene_names, cats,
               single_genes, multi_genes, AGE_COLORS,
               panel='polygon',
               verts_aligned=verts_aligned, cat_at_vertex=cat_at_vertex)

    plt.suptitle(
        'Gene Specificity across Developmental Ages  |  ILR -> PCA -> polygon alignment\n'
        'Circles = single-age specific (top 10 per age)   '
        'Triangles = multi-age specific (top 15 pairwise/triple)',
        fontsize=11, y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'ilr_pca_polygon.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out}")

    rows = []
    for cat in cats:
        for rank, idx in enumerate(single_genes[cat]):
            rows.append({
                'gene':  gene_names[idx],
                'group': f'single_{cat}',
                'rank_in_group': rank + 1,
                'psi_max_age': cat,
                'psi_score': round(float(W_norm[idx, cats.index(cat)]), 4),
            })
    for rank, idx in enumerate(multi_genes):
        top2_ages = [cats[j] for j in np.argsort(W_norm[idx])[::-1][:2]]
        rows.append({
            'gene':  gene_names[idx],
            'group': 'multi_' + '+'.join(top2_ages),
            'rank_in_group': rank + 1,
            'psi_max_age': top2_ages[0],
            'psi_score': round(float(W_norm[idx, cats.index(top2_ages[0])]), 4),
        })
    df_genes = pd.DataFrame(rows)
    gene_csv = os.path.join(OUTPUT_DIR, 'highlighted_genes.csv')
    df_genes.to_csv(gene_csv, index=False)
    print(f"Gene table saved to {gene_csv}")

    pd.DataFrame(pca_coords, index=gene_names,
                 columns=['PC1', 'PC2']).to_csv(
        os.path.join(OUTPUT_DIR, 'ilr_pca_coords.csv'))
    print("PCA coords saved.")


if __name__ == '__main__':
    main()
