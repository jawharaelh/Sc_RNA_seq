"""

"""
import os
import math
from itertools import permutations, combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-safe: must be set before pyplot is imported
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _ilr_transform(X):
    """
    Isometric Log-Ratio (ILR) transform, sequential-binary partition.
    Maps (n, K) compositional rows (sum to 1) to (n, K-1) Euclidean coords,
    so Euclidean methods like PCA are no longer geometrically distorted by
    the simplex constraint.
    """
    K = X.shape[1]
    X = np.clip(X, 1e-9, None)
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    ilr = np.zeros((X.shape[0], K - 1))
    for i in range(1, K):
        ilr[:, i - 1] = np.sqrt(i / (i + 1)) * (log_X[:, :i].mean(axis=1) - log_X[:, i])
    return ilr


def _procrustes_rotation(A, B):
    """
    Align point cloud A to point cloud B (rotation + uniform scale, SVD-based,
    sign-corrected against reflection). Returns A_aligned, R (2x2), s (float),
    residual (float MSE).
    """
    A_c = A - A.mean(0)
    B_c = B - B.mean(0)
    U, S, Vt = np.linalg.svd(B_c.T @ A_c)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    s = float((B_c * (A_c @ R.T)).sum() / (A_c ** 2).sum())
    A_aligned = s * A_c @ R.T + B.mean(0)
    residual = float(np.mean((A_aligned - B) ** 2))
    return A_aligned, R, s, residual


def _best_vertex_assignment(pca_coords, single_genes, categories, K):
    """
    Exhaustive search over all K! category-to-vertex permutations. For each,
    Procrustes-aligns the K polygon vertices to the K per-category PCA
    centroids and keeps the permutation with the lowest residual. Runtime is
    combinatorial in K (fine through K~9-10; needs a smarter search beyond that).
    """
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    pca_centroids = np.array([pca_coords[single_genes[cat]].mean(0) for cat in categories])

    best_perm, best_res = list(range(K)), np.inf
    for perm in permutations(range(K)):
        inv_perm = [0] * K
        for vi, ci in enumerate(perm):
            inv_perm[ci] = vi
        poly_centroids = np.array([vertices[inv_perm[j]] for j in range(K)])
        _, _, _, res = _procrustes_rotation(poly_centroids, pca_centroids)
        if res < best_res:
            best_res, best_perm = res, list(perm)
    return best_perm, best_res


def _select_highlight_genes(W_norm, gene_names, categories, n_single, n_per_pair, min_psi):
    """
    Group 1 - single-category specific: top n_single genes per category by
    normalized Psi_block score (each gene used at most once).
    Group 2 - multi-category specific: for every category pair, top
    n_per_pair genes by summed Psi_block score, excluding Group 1 genes.
    """
    used = set()

    single = {}
    for j, cat in enumerate(categories):
        order = np.argsort(W_norm[:, j])[::-1]
        selected, fallback = [], []
        for idx in order:
            if idx in used:
                continue
            if W_norm[idx, j] >= min_psi and len(selected) < n_single:
                selected.append(int(idx))
            elif W_norm[idx, j] < min_psi and len(fallback) < n_single:
                fallback.append(int(idx))
            if len(selected) == n_single:
                break
        if len(selected) < n_single:
            selected += fallback[:n_single - len(selected)]
        for idx in selected:
            used.add(int(idx))
        single[cat] = selected

    multi, multi_pairs, used_multi = [], {}, set()
    for i, j in combinations(range(len(categories)), 2):
        pair_scores = [
            (idx, float(W_norm[idx, i] + W_norm[idx, j]))
            for idx in range(len(gene_names))
            if idx not in used and idx not in used_multi
        ]
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in pair_scores[:n_per_pair]:
            multi.append(idx)
            multi_pairs[idx] = (categories[i], categories[j])
            used_multi.add(idx)

    return single, multi, multi_pairs


def _draw_panel(ax, coords, gene_names, categories, single_genes, multi_genes,
                 category_colors, panel, vertices_aligned=None, cat_at_vertex=None,
                 W_norm=None, var_exp=None, multi_pairs=None):
    """Shared drawing logic for the PCA panel and the aligned-polygon panel."""
    ax.scatter(coords[:, 0], coords[:, 1],
               c='lightgray', s=10, alpha=0.4, zorder=1, rasterized=True)

    for cat in categories:
        idxs = single_genes[cat]
        col = category_colors.get(cat, '#333333')
        ax.scatter(coords[idxs, 0], coords[idxs, 1], color=col, s=55, marker='o',
                   edgecolor='k', lw=0.4, zorder=3, label=f'{cat}')
        for idx in idxs[:3]:
            ax.annotate(gene_names[idx], (coords[idx, 0], coords[idx, 1]),
                        fontsize=6, color=col, xytext=(4, 4), textcoords='offset points')

    multi_by_cat = defaultdict(list)
    for idx in multi_genes:
        cat_a, cat_b = multi_pairs[idx]
        j_a, j_b = categories.index(cat_a), categories.index(cat_b)
        dom = cat_a if W_norm[idx, j_a] >= W_norm[idx, j_b] else cat_b
        multi_by_cat[dom].append(idx)

    labeled = set()
    for cat in categories:
        idxs_m = multi_by_cat.get(cat, [])
        if not idxs_m:
            continue
        col = category_colors.get(cat, '#333333')
        label = f'{cat} (multi)' if cat not in labeled else '_nolegend_'
        labeled.add(cat)
        ax.scatter(coords[idxs_m, 0], coords[idxs_m, 1], color=col, s=65, marker='^',
                   edgecolor='k', lw=0.4, zorder=4, alpha=0.75, label=label)
    for idx in multi_genes[:5]:
        dom_cat = next(c for c in categories if idx in multi_by_cat.get(c, []))
        ax.annotate(gene_names[idx], (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, color=category_colors.get(dom_cat, '#555555'),
                    xytext=(4, 4), textcoords='offset points')

    if panel == 'pca':
        span = float(np.ptp(coords, axis=0).max()) * 0.38
        for j, cat in enumerate(categories):
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
        ax.set_title('(a) PCA of ILR(Psi_block) — accurate geometry', fontsize=11)
    else:
        outline = np.vstack([vertices_aligned, vertices_aligned[0]])
        ax.plot(outline[:, 0], outline[:, 1], 'k-', lw=1.8, alpha=0.5, zorder=2)
        center = vertices_aligned.mean(0)
        for i, (x, y) in enumerate(vertices_aligned):
            cat_v = cat_at_vertex[i]
            col_v = category_colors.get(cat_v, 'k')
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


def plot_polygon_specificity(
    partition_label,
    psi_block_dir,
    pvals_dir,
    save_dir,
    categories=None,
    category_colors=None,
    psi_thresh=0.6,
    p_thresh=0.05,
    q_thresh=0.05,
    n_single=10,
    n_per_pair=2,
    min_psi=0.65,
    gene_symbol_lookup=None,
):
    """
    Generate an ILR -> PCA -> Procrustes-aligned polygon specificity plot.

    Psi_block rows are compositional (each gene's row sums to 1 across
    categories), so naively placing genes on a regular polygon via
    barycentric coordinates distorts real geometric relationships between
    genes. This function instead: (1) filters to statistically significant
    genes, (2) row-normalizes Psi_block and applies an ILR transform to map
    the simplex into true Euclidean space, (3) runs PCA on the ILR
    coordinates as the "ground truth" gene geometry, (4) picks
    single-category- and pairwise-category-specific highlight genes, and
    (5) searches all K! category-to-vertex assignments, Procrustes-aligning
    the polygon vertices to each category's PCA centroid, keeping the
    best-fitting permutation/rotation/scale. Draws both the PCA panel and
    the aligned polygon panel side by side, so the polygon is not
    geometrically arbitrary.

    Parameters
    ----------
    partition_label : str, Required
        The partition label used to find input files and label outputs
        (e.g. 'Genotype', 'Age'). Matches what was passed to light_ember.

    psi_block_dir : str, Required
        Path to the directory containing 'mean_Psi_block_df_{partition_label}.csv'
        (the Psi_block_df folder produced by light_ember).

    pvals_dir : str, Required
        Path to the CSV containing p-values (as produced by generate_pvals),
        used to filter to statistically significant genes before plotting.
        Must include 'Psi', 'Psi p-value', and 'Psi q-value' columns.

    save_dir : str, Required
        Path to directory where the plot and gene tables will be saved.

    categories : list[str], default=None
        Order of categories, used for color assignment / display only.
        Defaults to the column order in the Psi_block file. Does NOT affect
        the optimal vertex assignment, which is chosen by geometry
        regardless of this order.

    category_colors : dict, default=None
        {category: hex color}. Defaults to a qualitative colormap ('tab10'
        for K<=10, else 'tab20').

    psi_thresh : float, default=0.6
        Minimum overall Psi to include a gene at all.

    p_thresh : float, default=0.05
        Maximum 'Psi p-value' to include a gene.

    q_thresh : float, default=0.05
        Maximum 'Psi q-value' to include a gene.

    n_single : int, default=10
        Number of top single-category-specific genes to highlight per category.

    n_per_pair : int, default=2
        Number of top pairwise-category-specific genes to highlight per
        category pair.

    min_psi : float, default=0.65
        Minimum normalized Psi_block score for a gene to qualify as
        single-category-specific (falls back to next-best genes per
        category if too few clear that bar).

    gene_symbol_lookup : dict or pandas.Series, default=None
        Optional {gene_id: gene_symbol} mapping for readable plot labels,
        e.g. when the Psi_block index is Ensembl IDs rather than gene
        symbols. Defaults to using the index as-is.

    Returns
    -------
    None

    Notes
    -----
    - Saves 'polygon_specificity_{partition_label}.png'
    - Saves 'polygon_highlighted_genes_{partition_label}.csv'
    - Saves 'polygon_pca_coords_{partition_label}.csv'
    - The K! vertex search is brute-force: fine through K~9-10 categories
      (< 1M permutations), but needs a smarter search (e.g. greedy or
      simulated annealing) for substantially larger K.
    """
    psi_block_dir = os.path.expanduser(psi_block_dir)
    pvals_dir = os.path.expanduser(pvals_dir)
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    psi_path = os.path.join(psi_block_dir, f'mean_Psi_block_df_{partition_label}.csv')
    df_psi = pd.read_csv(psi_path, index_col=0)

    if categories is None:
        categories = df_psi.columns.tolist()
    else:
        missing = set(categories) - set(df_psi.columns)
        if missing:
            raise ValueError(
                f"categories {sorted(missing)} not found in {psi_path}. "
                f"Available: {list(df_psi.columns)}"
            )
    K = len(categories)
    df_psi = df_psi[categories].dropna()

    if category_colors is None:
        cmap = plt.get_cmap('tab10' if K <= 10 else 'tab20')
        category_colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

    df_pvals = pd.read_csv(pvals_dir, index_col=0)
    sig_genes = df_pvals.index[
        (df_pvals['Psi p-value'] < p_thresh) &
        (df_pvals['Psi q-value'] < q_thresh) &
        (df_pvals['Psi'] > psi_thresh)
    ]
    df_top = df_psi.loc[df_psi.index.intersection(sig_genes)]
    print(f"Genes passing p<{p_thresh}, q<{q_thresh}, Psi>{psi_thresh}: {len(df_top)}")
    if len(df_top) < K:
        raise ValueError(
            "Fewer significant genes than categories — psi_thresh/p_thresh/q_thresh "
            "are likely too strict for this data."
        )

    gene_names = df_top.index.tolist()
    if gene_symbol_lookup is not None:
        gene_names = [gene_symbol_lookup.get(g, g) for g in gene_names]

    W = df_top.values.astype(float)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-9)

    ilr = _ilr_transform(W_norm)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(ilr)
    var_exp = pca.explained_variance_ratio_ * 100
    print(f"ILR: {len(gene_names)} genes x {K} categories -> {K - 1} coords; "
          f"PCA: PC1={var_exp[0]:.1f}% PC2={var_exp[1]:.1f}%")

    single_genes, multi_genes, multi_pairs = _select_highlight_genes(
        W_norm, gene_names, categories, n_single, n_per_pair, min_psi)

    print(f"Searching {math.factorial(K)} vertex permutations...")
    best_perm, res = _best_vertex_assignment(pca_coords, single_genes, categories, K)
    cat_at_vertex = [categories[best_perm[i]] for i in range(K)]
    print(f"Best vertex order: {cat_at_vertex} (residual={res:.5f})")

    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    inv_perm = [0] * K
    for vi, ci in enumerate(best_perm):
        inv_perm[ci] = vi

    pca_centroids = np.array([pca_coords[single_genes[cat]].mean(0) for cat in categories])
    poly_centroids = np.array([vertices[inv_perm[j]] for j in range(K)])
    _, R, s, _ = _procrustes_rotation(poly_centroids, pca_centroids)
    cent_mean = poly_centroids.mean(0)
    pca_mean = pca_centroids.mean(0)

    poly_coords = W_norm[:, best_perm] @ vertices
    poly_aligned = s * (poly_coords - cent_mean) @ R.T + pca_mean
    vertices_aligned = s * (vertices - cent_mean) @ R.T + pca_mean

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    _draw_panel(axes[0], pca_coords, gene_names, categories, single_genes, multi_genes,
                category_colors, panel='pca', W_norm=W_norm, var_exp=var_exp, multi_pairs=multi_pairs)
    _draw_panel(axes[1], poly_aligned, gene_names, categories, single_genes, multi_genes,
                category_colors, panel='polygon', vertices_aligned=vertices_aligned,
                cat_at_vertex=cat_at_vertex, W_norm=W_norm, multi_pairs=multi_pairs)

    plt.suptitle(
        f'Gene Specificity across {partition_label} | ILR -> PCA -> polygon alignment\n'
        f'Circles = single-category specific (top {n_single} per category)   '
        f'Triangles = multi-category specific (top {n_per_pair} per pair)',
        fontsize=11, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'polygon_specificity_{partition_label}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Polygon specificity plot saved to {out_path}')

    rows = []
    for cat in categories:
        for rank, idx in enumerate(single_genes[cat]):
            rows.append({
                'gene': gene_names[idx], 'group': f'single_{cat}', 'rank_in_group': rank + 1,
                'psi_max_category': cat, 'psi_score': round(float(W_norm[idx, categories.index(cat)]), 4),
            })
    for rank, idx in enumerate(multi_genes):
        pair = multi_pairs[idx]
        rows.append({
            'gene': gene_names[idx], 'group': f'multi_{pair[0]}+{pair[1]}', 'rank_in_group': rank + 1,
            'psi_max_category': pair[0], 'psi_score': round(float(W_norm[idx, categories.index(pair[0])]), 4),
        })
    genes_path = os.path.join(save_dir, f'polygon_highlighted_genes_{partition_label}.csv')
    pd.DataFrame(rows).to_csv(genes_path, index=False)
    print(f'Highlighted gene table saved to {genes_path}')

    coords_path = os.path.join(save_dir, f'polygon_pca_coords_{partition_label}.csv')
    pd.DataFrame(pca_coords, index=gene_names, columns=['PC1', 'PC2']).to_csv(coords_path)
    print(f'PCA coords saved to {coords_path}')
