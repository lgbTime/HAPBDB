#!/data/lgb/software/anaconda3/bin/python
import matplotlib
matplotlib.use('Agg')

# --------------------------------
# Journal-grade figure styling
# --------------------------------
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["font.size"] = 8
rcParams["axes.titlesize"] = 10
rcParams["axes.labelsize"] = 9
rcParams["xtick.labelsize"] = 7
rcParams["ytick.labelsize"] = 7
rcParams["legend.fontsize"] = 8
rcParams["axes.linewidth"] = 0.6
rcParams["xtick.major.width"] = 0.6
rcParams["ytick.major.width"] = 0.6
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import argparse
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import os
try:
    import cairo
    CAIRO_BACKEND = "cairo"
except ImportError:
    CAIRO_BACKEND = "pdf"

# Colorblind-safe palette for dendrogram cluster branches (Okabe-Ito + ColorBrewer Dark2/Set2 extensions)
LINK_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9',
               '#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02',
               '#A6761D', '#8DA0CB', '#E78AC3', '#A6D854', '#FC8D62', '#66C2A5',
               '#B3B3B3', '#666666']
ABOVE_THRESHOLD_COLOR = '#333333'


def _link_color_func(linkage_matrix, n_samples, threshold, palette=LINK_COLORS):
    """Build a scipy dendrogram link_color_func coloring each link by its flat
    cluster at `threshold` (identical to the *_extram_based.txt membership).
    Links at/above the threshold (tree trunk) are dark gray. Returns None when
    `threshold` is None, letting scipy use its default cluster coloring."""
    if threshold is None:
        return None
    flat = sch.fcluster(linkage_matrix, threshold, criterion='distance')
    n_links = len(linkage_matrix)
    link_cluster = [None] * n_links
    for i in range(n_links):
        if linkage_matrix[i, 2] >= threshold:
            continue
        # any leaf under this link belongs to its flat cluster
        stack = [n_samples + i]
        first_leaf = None
        while stack:
            node = stack.pop()
            if node < n_samples:
                first_leaf = node
                break
            stack.extend([int(linkage_matrix[node - n_samples, 0]),
                          int(linkage_matrix[node - n_samples, 1])])
        link_cluster[i] = flat[first_leaf]

    def _color(link_id):
        # scipy passes extended ids: leaves 0..n-1, links n..2n-2
        row = link_id - n_samples
        if row < 0 or row >= n_links:
            return ABOVE_THRESHOLD_COLOR
        cl = link_cluster[row]
        if cl is None:
            return ABOVE_THRESHOLD_COLOR
        return palette[(cl - 1) % len(palette)]

    return _color

### Construct haplotype tree and perform clustering analysis based on hierarchical clustering
def treebase_hap(distance_matrix, sample_ids, traits, prefix):
    """traits: list of (name, e1, e2). Returns dict[name -> {"threshold": float|None, "cluster_map": dict|None}]."""
    hpix = int(len(sample_ids) / 12)
    sample_ids = np.array(sample_ids)
    # linkage expects a condensed distance vector; passing the full symmetric
    # matrix silently corrupts the merge heights (scipy <= 1.15 warning only)
    linkage_matrix = sch.linkage(squareform(distance_matrix, checks=False), method='average')
    np.save(f"{prefix}_linkage_matrix.npy", linkage_matrix)
    print(f"linkage_matrix saved to {prefix}_linkage_matrix.npy")
    with open(f"{prefix}_sample_ids.txt", "w") as fout:
        for sid in sample_ids:
            fout.write(sid + "\n")

    # Distance=1 clustering (baseline, shared across traits)
    with open(f"{prefix}_tree_base_dis1_haplotype.txt", 'w') as tree_dist1:
        clusters = sch.fcluster(linkage_matrix, 1, criterion='distance')
        for cluster_id in np.unique(clusters):
            cluster_samples = sample_ids[clusters == cluster_id]
            tree_dist1.write(f"haplotype_by_tree_distance_1:\thaplotype:{cluster_id}:\t{','.join(cluster_samples)}\n")

    trait_results = {}
    active_traits = [(n, e1, e2) for n, e1, e2 in traits if e1 is not None and e2 is not None]
    if not active_traits:
        print("[i] no trait accessions provided, skip dynamic threshold separation.")
        return trait_results

    for trait_name, e1, e2 in active_traits:
        if e1 not in sample_ids or e2 not in sample_ids:
            print(f"Error: {e1} or {e2} not in sample_ids, skipping trait '{trait_name}'.")
            trait_results[trait_name] = {"threshold": None, "cluster_map": None}
            continue

        found = False
        for threshold_distance in np.arange(1.0, 0.01, -0.01):
            clusters = sch.fcluster(linkage_matrix, threshold_distance, criterion='distance')
            e1_cluster, e2_cluster = None, None
            for cluster_id in np.unique(clusters):
                cluster_samples = sample_ids[clusters == cluster_id]
                if e1 in cluster_samples:
                    e1_cluster = cluster_id
                if e2 in cluster_samples:
                    e2_cluster = cluster_id
                if e1_cluster is not None and e2_cluster is not None and e1_cluster != e2_cluster:
                    threshold_rounded = round(threshold_distance, 2)
                    print(f"[{trait_name}] At threshold {threshold_rounded}, {e1} and {e2} are in different clusters.")
                    with open(f"{prefix}_{trait_name}_tree_dist{threshold_rounded}_extram_based.txt", 'w') as tree_base_out_file:
                        for cid in np.unique(clusters):
                            cs = sample_ids[clusters == cid]
                            tree_base_out_file.write(f"H{cid}:\t{','.join(cs)}\n")
                    trait_results[trait_name] = {
                        "threshold": threshold_distance,
                        "cluster_map": {sid: clusters[i] for i, sid in enumerate(sample_ids)}
                    }
                    trait_results[trait_name]["scores"] = compute_cophenetic_scores(
                        linkage_matrix, sample_ids, e1, e2)
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"[!] {trait_name}: no threshold found separating {e1} from {e2}")
            trait_results[trait_name] = {"threshold": None, "cluster_map": None}
            trait_results[trait_name]["scores"] = compute_cophenetic_scores(
                linkage_matrix, sample_ids, e1, e2)

    # Dendrogram with branches colored by cluster: below the separation threshold each
    # cluster gets its own color, matching the *_extram_based.txt cluster membership file.
    sep_threshold = next((tr["threshold"] for tr in trait_results.values() if tr.get("threshold") is not None), None)
    sch.set_link_color_palette(LINK_COLORS)
    fig, ax = plt.subplots(figsize=(10, hpix))
    sch.dendrogram(linkage_matrix, labels=sample_ids, orientation='left', leaf_rotation=360,
                   leaf_font_size=6, color_threshold=sep_threshold,
                   above_threshold_color=ABOVE_THRESHOLD_COLOR,
                   link_color_func=_link_color_func(linkage_matrix, len(sample_ids), sep_threshold),
                   ax=ax)
    ax.set_xlabel('Genetic distance (Hamming)')
    ax.set_ylabel('Sample')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    with PdfPages(f"{prefix}_tree_dendrogram.pdf") as pdf_pages:
        pdf_pages.savefig()
    plt.close(fig)

    return trait_results

### Compute cophenetic-distance-based scores for each sample.
### Score is based on proximity to e1 (favorable), with e2 (unfavorable) anchoring the negative end.
def compute_cophenetic_scores(linkage_matrix, sample_ids, e1, e2):
    """Return dict mapping sample_id -> integer score in [-100, 100].

    For each sample s, d1=cophenetic distance to e1, d2=to e2, d12=between e1 and e2.
    - Closer to e1 than e2 (d1 < d2): score = 100 * (1 - d1/d12), clamped to [0, 100]
    - Closer to e2 than e1 (d2 < d1): score = -100 * (1 - d2/d12), clamped to [-100, 0]
    - Equidistant (d1 == d2): score = 0
    Rounded to nearest 10.
    """
    n = len(sample_ids)
    idx1 = list(sample_ids).index(e1)
    idx2 = list(sample_ids).index(e2)

    # Build condensed cophenetic distance matrix from linkage
    nn = n * (n - 1) // 2
    coph = np.zeros(nn)
    clusters = [{i} for i in range(n)]
    for row in linkage_matrix:
        i, j = int(row[0]), int(row[1])
        dist = row[2]
        for a in clusters[i]:
            for b in clusters[j]:
                if a < b:
                    p = n * a - a * (a + 1) // 2 + (b - a - 1)
                else:
                    p = n * b - b * (b + 1) // 2 + (a - b - 1)
                coph[p] = dist
        clusters.append(clusters[i] | clusters[j])

    def coph_dist(i, j):
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        return coph[n * i - i * (i + 1) // 2 + (j - i - 1)]

    d12 = coph_dist(idx1, idx2)
    if d12 == 0:
        return {s: 0 for s in sample_ids}

    root_height = max(row[2] for row in linkage_matrix)

    scores = {}
    for k, s in enumerate(sample_ids):
        d1 = coph_dist(k, idx1)
        d2 = coph_dist(k, idx2)
        if d1 < d2:
            # Favorable side: score decays from 100 toward 0 as d1 approaches d12
            raw = 100.0 * (1.0 - d1 / d12)
            raw = max(0.0, min(100.0, raw))
        elif d2 < d1:
            # Unfavorable side: score decays from -100 toward 0 as d2 approaches d12
            raw = -100.0 * (1.0 - d2 / d12)
            raw = max(-100.0, min(0.0, raw))
        else:
            # Equidistant — penalize distance beyond d12
            if d1 <= d12:
                raw = 0.0
            else:
                # Far from both references: negative, proportional to excess distance
                excess = (d1 - d12) / max(root_height - d12, 1e-10)
                raw = -100.0 * excess
        scores[s] = int(round(raw / 10) * 10)
    return scores

### Convert VCF file to one-hot encoded haplotype array.
def vcf_to_haplotype_array_one_hot(vcf_file):
    haplotype_dict = {}
    sample_ids = []
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    sample_ids = line.strip().split('\t')[9:]
                continue
            columns = line.strip().split('\t')
            genotypes = columns[9:]
            for sample_index, genotype in enumerate(genotypes):
                sample_id = sample_ids[sample_index]
                if sample_id not in haplotype_dict:
                    haplotype_dict[sample_id] = []
                if genotype == '0/0':
                    haplotype_dict[sample_id].append([1, 0, 0])
                elif genotype == '1/1':
                    haplotype_dict[sample_id].append([0, 1, 0])
                elif genotype in ['0/1', '1/0']:
                    haplotype_dict[sample_id].append([0, 0, 1])
                elif genotype in ['.', './.']:
                    haplotype_dict[sample_id].append([0, 0, 0])
                else:
                    haplotype_dict[sample_id].append([0, 0, 0])
    haplotypes_array = np.array([np.array(haplotype_dict[sample_id]).flatten() for sample_id in haplotype_dict])
    return haplotypes_array, sample_ids

### Perform unsupervised clustering on haplotypes from VCF file.
def uclu(pop_vcf, prefix):
    haplotypes_array, sample_ids = vcf_to_haplotype_array_one_hot(pop_vcf)
    unique_haplotypes, unique_indices = np.unique(haplotypes_array, axis=0, return_inverse=True)
    n_clusters = int(len(unique_haplotypes))
    distance_matrix_unique = pairwise_distances(unique_haplotypes, metric='hamming')
    distance_matrix_all = pairwise_distances(haplotypes_array, metric='hamming')
    np.save(f"{prefix}_distance_matrix.npy", distance_matrix_all)
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    clusters = agg_cluster.fit_predict(distance_matrix_unique)
    sample_clusters = pd.DataFrame({
        'sampleID': sample_ids,
        'cluster': clusters[unique_indices]
    })
    sample_clusters.to_csv(f"{prefix}_strict_all_hap.txt", header=True, sep="\t", index=None)
    return sample_clusters, n_clusters, distance_matrix_all, unique_indices, sample_ids, haplotypes_array


### Generate and save a heatmap visualization of pairwise distances.
def plot_heatmap(distance_matrix, sample_ids, filename):
    """Saves two versions: an unlabeled publication-style heatmap and a labeled
    version carrying small sample-ID labels on both axes."""
    n = len(sample_ids)
    base = filename.replace('.pdf', '')
    # --- Unlabeled version (publication style) ---
    fig, ax = plt.subplots(figsize=(min(9.0, max(4.5, n / 12)), min(9.0, max(4.5, n / 12))))
    # Hide per-sample tick labels at population scale (unreadable when overlapping);
    # the colorbar alone carries the quantitative information.
    sns.heatmap(distance_matrix, xticklabels=False, yticklabels=False, cmap='YlGnBu',
                square=True, linewidths=0, ax=ax, cbar_kws={'label': 'Hamming distance', 'shrink': 0.8})
    ax.set_title('Pairwise Hamming Distance', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    with PdfPages(f"{base}.pdf") as pdf_pages:
        pdf_pages.savefig()
    plt.close(fig)
    # --- Labeled version (small sample-ID labels on both axes) ---
    label_fs = max(3, min(7, int(420 / n)))
    fig, ax = plt.subplots(figsize=(min(12.0, max(6.0, n / 8)), min(12.0, max(6.0, n / 8))))
    sns.heatmap(distance_matrix, xticklabels=sample_ids, yticklabels=sample_ids, cmap='YlGnBu',
                square=True, linewidths=0, ax=ax, cbar_kws={'label': 'Hamming distance', 'shrink': 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=label_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fs)
    ax.set_title('Pairwise Hamming Distance (labeled)', pad=10)
    fig.tight_layout()
    with PdfPages(f"{base}_labeled.pdf") as pdf_pages:
        pdf_pages.savefig()
    plt.close(fig)
    print(f"[+] Heatmaps saved to {base}.pdf and {base}_labeled.pdf")


### Generate a PCA scatter plot of haplotypes colored by cluster, with e1/e2 highlighted.
def plot_pca(haplotypes_array, sample_ids, cluster_map, acc1, acc2, prefix):
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pca = PCA(n_components=2)
    coords = pca.fit_transform(haplotypes_array)
    var = pca.explained_variance_ratio_ * 100

    labels = [str(cluster_map.get(sid, 'H?')) for sid in sample_ids] if cluster_map else ['H1'] * len(sample_ids)
    unique_labels = sorted(set(labels))
    color_map = {lab: LINK_COLORS[i % len(LINK_COLORS)] for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lab in unique_labels:
        mask = np.array(labels) == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=30, c=color_map[lab], label=lab,
                   edgecolors='white', linewidths=0.4, alpha=0.9, zorder=2)

    # Highlight the extreme accessions with distinct markers + ID labels
    e1_marker, e2_marker = '*', '^'
    for acc, marker, color in [(acc1, e1_marker, '#D55E00'), (acc2, e2_marker, '#0072B2')]:
        if acc is None or acc not in sample_ids:
            continue
        idx = list(sample_ids).index(acc)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=170, marker=marker, c=color,
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(acc, (coords[idx, 0], coords[idx, 1]), xytext=(9, 9),
                    textcoords='offset points', fontsize=8, fontweight='bold', color='black')

    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax.set_title('Haplotype PCA', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [Patch(facecolor=color_map[lab], edgecolor='none', label=lab) for lab in unique_labels]
    if acc1 is not None or acc2 is not None:
        if acc1 is not None:
            handles.append(Line2D([0], [0], marker=e1_marker, linestyle='none', markersize=9,
                                  markerfacecolor='#D55E00', markeredgecolor='black',
                                  label='e1 (favorable)'))
        if acc2 is not None:
            handles.append(Line2D([0], [0], marker=e2_marker, linestyle='none', markersize=9,
                                  markerfacecolor='#0072B2', markeredgecolor='black',
                                  label='e2 (unfavorable)'))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=7, title='Cluster', title_fontsize=8)

    fig.tight_layout()
    plt.savefig(f"{prefix}_PCA.pdf", dpi=300, bbox_inches='tight', format='pdf', backend=CAIRO_BACKEND)
    plt.close(fig)
    print(f"[+] PCA plot saved to {prefix}_PCA.pdf (PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%)")


### Load a two-column phenotype file (sample_id, value).
### Auto-detects tab/space/comma separators and an optional header line.
def load_phenotype_file(path):
    import re
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'[\t,; ]+', line)
            if len(parts) < 2:
                continue
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                continue  # header line or non-numeric token
    if len(values) < 2:
        raise ValueError(f"phenotype file '{path}' must contain at least 2 'sample value' lines")
    return values


### Draw a per-haplotype-cluster boxplot of observed phenotypes. Only runs when
### a phenotype file is provided via --phenotype-file.
def plot_phenotype_boxplot(sample_ids, cluster_map, prefix, phenotype_label='phenotype',
                           observed=None, anchors=None, fallback_anchors=None, seed=42):
    """`observed`: {sample_id: float} phenotypes loaded from --phenotype-file.
    `anchors`/`fallback_anchors` name accessions to annotate on the plot (their
    observed values are shown); if none of them have values, the samples with
    the minimum and maximum phenotype are annotated instead.
    Writes {prefix}_{phenotype}_boxplot.pdf and {prefix}_{phenotype}_phenotype.txt."""
    if observed is None:
        print("[i] phenotype boxplot skipped: provide --phenotype-file to activate it.")
        return
    slug = ''.join(c if c.isalnum() else '_' for c in phenotype_label.lower()).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    labels = [str(cluster_map.get(sid, 'H?')) for sid in sample_ids] if cluster_map else ['H1'] * len(sample_ids)

    phenos = np.array([observed.get(sid, np.nan) for sid in sample_ids], dtype=float)
    n_obs = int(np.isfinite(phenos).sum())
    if n_obs < 3:
        print(f"[!] phenotype boxplot skipped: only {n_obs} samples with observed phenotypes")
        return

    # annotate requested accessions that have observed values; else min/max samples
    present = {}
    for acc in (anchors or {}):
        if acc in sample_ids and np.isfinite(phenos[sample_ids.index(acc)]):
            present[acc] = phenos[sample_ids.index(acc)]
    for acc in (fallback_anchors or {}):
        if acc in sample_ids and np.isfinite(phenos[sample_ids.index(acc)]) and acc not in present:
            present[acc] = phenos[sample_ids.index(acc)]
    if not present:
        valid = [(sid, p) for sid, p in zip(sample_ids, phenos) if np.isfinite(p)]
        lo = min(valid, key=lambda t: t[1])
        hi = max(valid, key=lambda t: t[1])
        present = {lo[0]: lo[1], hi[0]: hi[1]}

    # Group by haplotype cluster; order clusters by median phenotype
    keep = np.isfinite(phenos)
    order = sorted(set(np.array(labels)[keep]),
                   key=lambda lab: np.median(phenos[np.array(labels) == lab]))
    color_map = {lab: LINK_COLORS[i % len(LINK_COLORS)] for i, lab in enumerate(order)}

    fig, ax = plt.subplots(figsize=(max(5.0, len(order) * 0.9), 4.6))
    data = [phenos[(np.array(labels) == lab) & keep] for lab in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', linewidth=1.1),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor='#999999',
                                    markeredgecolor='none', alpha=0.6))
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    for patch, lab in zip(bp['boxes'], order):
        patch.set_facecolor(color_map[lab])
        patch.set_edgecolor('#333333')
        patch.set_linewidth(0.7)
    # Overlay individual samples with light jitter
    for i, lab in enumerate(order):
        vals = data[i]
        x = np.random.default_rng(seed + i).normal(i + 1, 0.06, len(vals))
        ax.scatter(x, vals, s=14, color='black', alpha=0.35, zorder=3, linewidths=0)
    # Annotate extreme/anchored accessions
    for acc, val in present.items():
        lab = str(cluster_map.get(acc, 'H?'))
        if lab in order:
            x = order.index(lab) + 1
            ax.annotate(f'{acc}\n({val:.0f} d)', (x, val), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=7, fontweight='bold')

    ax.set_ylabel(phenotype_label)
    ax.set_xlabel('Haplotype cluster')
    ax.set_title(f'{phenotype_label} by Haplotype Cluster', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig(f"{prefix}_{slug}_boxplot.pdf", dpi=300, bbox_inches='tight', format='pdf',
                backend=CAIRO_BACKEND)
    plt.close(fig)

    with open(f"{prefix}_{slug}_phenotype.txt", "w") as f:
        f.write("sample_id\tcluster\tphenotype\n")
        for sid, lab, p in zip(sample_ids, labels, phenos):
            if np.isfinite(p):
                f.write(f"{sid}\t{lab}\t{p:.2f}\n")
    print(f"[+] {phenotype_label} boxplot saved to {prefix}_{slug}_boxplot.pdf "
          f"(observed phenotypes, n={n_obs})")

### Map an allele to a display symbol: SNP base, 'I' (indel), 'SV' or 'NA'.
def classify_allele(a):
    if a in ('A', 'T', 'G', 'C'):
        return a
    if a == '*' or (a.startswith('<') and a.endswith('>')):
        return 'SV'
    if len(a) > 1:
        return 'I'
    return 'NA'


### Convert VCF file to a base-level matrix DataFrame.
def vcf_to_base_matrix(vcf_file):
    sample_ids = []
    variant_ids = []
    base_matrix = []
    with open(vcf_file) as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                sample_ids = line.strip().split('\t')[9:]
                continue
            cols = line.strip().split('\t')
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            alts = alt.split(',')
            variant_id = f"{chrom}_{pos}"
            variant_ids.append(variant_id)
            gts = []
            for s in cols[9:]:
                gt = s.split(":")[0]
                if gt in ['./.', '.']:
                    gts.append('NA')
                else:
                    try:
                        alleles = [ref if i == '0' else alts[int(i) - 1] for i in gt.replace('|', '/').split('/')]
                        gts.append('/'.join(classify_allele(a) for a in alleles))
                    except:
                        gts.append('NA')
            base_matrix.append(gts)
    base_df = pd.DataFrame(base_matrix, index=variant_ids, columns=sample_ids).T
    return base_df

### Generate and save a haplotype base table visualization with clustering information.
def plot_base_hap_table_from_clustering(base_df, sample_clusters, tree_cluster_map, acc1, acc2, output_file, max_reps=None):
    """tree_cluster_map: dict[sample_id -> cluster_label] or None (falls back to strict clustering)."""
    # Colorblind-safe (Okabe-Ito) allele colors, suitable for high-impact journals.
    # 'I' = indel, 'SV' = structural variant.
    base_colors = {
        'A': '#0072B2', 'T': '#D55E00', 'G': '#009E73', 'C': '#CC79A7',
        'I': '#E69F00', 'SV': '#8C564B', 'NA': '#BDBDBD'
    }
    legend_alleles = ['A', 'T', 'G', 'C', 'I', 'SV', 'NA']
    if tree_cluster_map is None:
        tree_cluster_map = {row['sampleID']: f"H{row['cluster']}" for _, row in sample_clusters.iterrows()}

    base_df_out = base_df.join(sample_clusters.set_index('sampleID'))
    base_df_out['tree_cluster'] = base_df_out.index.map(tree_cluster_map)
    base_df_out['clustered_id'] = [f"{row.tree_cluster}_{idx}" for idx, row in base_df_out.iterrows()]
    base_df_out.index = base_df_out['clustered_id']
    base_df_out = base_df_out.drop(columns=['clustered_id'])

    representatives = []
    for cluster_id in sorted(base_df_out['tree_cluster'].dropna().unique()):
        subset = base_df_out[base_df_out['tree_cluster'] == cluster_id]
        matched = []
        if acc1 is not None or acc2 is not None:
            for idx in subset.index:
                if (acc1 is not None and acc1 in idx) or (acc2 is not None and acc2 in idx):
                    matched.append(idx)
        if matched:
            representatives.extend(matched)
        else:
            representatives.append(subset.index[0])
    # Collect sample indices containing acc1 or acc2 first (if provided)
    acc_matches = []
    for idx in base_df_out.index:
        if (acc1 is not None and acc1 in idx) or (acc2 is not None and acc2 in idx):
            acc_matches.append(idx)
    representatives = list(dict.fromkeys(acc_matches + representatives))[:max_reps]
    filtered_df = base_df_out.loc[representatives].drop(columns=['cluster', 'tree_cluster'])
    filtered_df = filtered_df.loc[:, (filtered_df != filtered_df.iloc[0]).any()]
    filtered_df = filtered_df.loc[:, ~(filtered_df == 'NA').any()]
    base_df.to_csv(f"{output_file.replace('.pdf', '_full_table.txt')}", sep='\t')
    fig, ax = plt.subplots(figsize=(max(8, len(filtered_df.columns)*0.35), max(4, len(filtered_df)*0.4)))
    for i, sample in enumerate(filtered_df.index):
        for j, variant in enumerate(filtered_df.columns):
            val = filtered_df.loc[sample, variant]
            if val == 'NA':
                color = base_colors['NA']
                display = 'NA'
            else:
                bases = val.split('/')
                display = ''.join(sorted(set(bases)))
                color = base_colors.get(bases[0], '#ffffff')
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', lw=0.4))
            ax.text(j+0.5, i+0.5, display, ha='center', va='center', fontsize=7)
    ax.set_xticks([i + 0.5 for i in range(len(filtered_df.columns))])
    ax.set_xticklabels(filtered_df.columns, rotation=90, fontsize=6, ha='center')
    ax.set_yticks([i + 0.5 for i in range(len(filtered_df.index))])
    ax.set_yticklabels(filtered_df.index, fontsize=6)
    ax.set_xlim(0, len(filtered_df.columns))
    ax.set_ylim(0, len(filtered_df.index))
    ax.invert_yaxis()
    ax.set_title("Haplotype Base Table", pad=10)
    ax.spines[:].set_visible(False)
    # Allele color legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=base_colors[b], edgecolor='none', label=b) for b in legend_alleles]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.03),
              ncol=len(legend_alleles), frameon=False, fontsize=8, handlelength=1.0, handleheight=1.0)
    fig.tight_layout()
    plt.savefig(output_file, dpi=300, format='pdf', bbox_inches='tight', backend=CAIRO_BACKEND)
    print(f"[+] Haplotype base plot saved to: {output_file}")

### Generate and save a combined visualization of dendrogram and haplotype base table.
def plot_tree_with_base_table(linkage_matrix_file, hap_base_file, output_pdf, prefix="result", color_threshold=None):
    import matplotlib.gridspec as gridspec
    # Read linkage matrix
    linkage_matrix = np.load(linkage_matrix_file, allow_pickle=True)

    # Read haplotype base full table
    base_df = pd.read_csv(hap_base_file, sep='\t', index_col=0, dtype=str).fillna('NA')
    base_df = base_df.loc[:, ~(base_df == 'NA').any()]

    # Dynamically adjust figure size
    n_samples, n_variants = base_df.shape
    row_height = 0.3
    col_width = 0.25
    fig_width = 4 + n_variants * col_width
    fig_height = max(6, n_samples * row_height)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    # --- Left panel: dendrogram ---
    ax1 = plt.subplot(gs[0])
    sch.set_link_color_palette(LINK_COLORS)
    dendro = sch.dendrogram(linkage_matrix, orientation='left', labels=base_df.index,
                            color_threshold=color_threshold, above_threshold_color=ABOVE_THRESHOLD_COLOR,
                            link_color_func=_link_color_func(linkage_matrix, len(base_df), color_threshold),
                            ax=ax1)
    leaf_order = dendro['ivl']   # Get leaf order from dendrogram
    leaf_order_for_table = leaf_order[::-1]

    # --- Right panel: base table ---
    ordered_df = base_df.reindex(leaf_order_for_table)
    ax2 = plt.subplot(gs[1])

    base_colors = {
        'A': '#0072B2', 'T': '#D55E00', 'G': '#009E73',
        'C': '#CC79A7', 'I': '#E69F00', 'SV': '#8C564B', 'NA': '#BDBDBD'
    }
    legend_alleles = ['A', 'T', 'G', 'C', 'I', 'SV', 'NA']

    for i, sample in enumerate(ordered_df.index):
        for j, val in enumerate(ordered_df.columns):
            base = ordered_df.loc[sample, val]
            if base == 'NA':
                color, text = base_colors['NA'], 'NA'
            else:
                bases = str(base).split('/')
                text = ''.join(sorted(set(bases)))
                color = base_colors.get(bases[0], '#ffffff')
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', lw=0.4))
            ax2.text(j+0.5, i+0.5, text, ha='center', va='center', fontsize=5)

    ax2.set_xlim(0, len(ordered_df.columns))
    ax2.set_ylim(0, len(ordered_df.index))
    ax2.invert_yaxis()
    ax2.set_xticks(np.arange(len(ordered_df.columns)) + 0.5)
    ax2.set_xticklabels(ordered_df.columns, rotation=90, fontsize=5, ha='center')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.spines[:].set_visible(False)

    # Dendrogram styling
    ax1.set_xlabel('Genetic distance (Hamming)', fontsize=7)
    ax1.tick_params(labelsize=5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Allele color legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=base_colors[b], edgecolor='none', label=b) for b in legend_alleles]
    ax2.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(legend_alleles), frameon=False, fontsize=7, handlelength=1.0, handleheight=1.0)

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[+] Combined tree + haplotype base table saved to {output_pdf}")

### Build a breeding selection matrix from per-trait cophenetic scores.
def build_breeding_matrix(sample_ids_all, trait_results):
    """Returns DataFrame: rows=samples, columns=[{trait}_score, ...], sorted by total_score descending."""
    # Collect all samples that appear in any trait's scores
    all_scored = set()
    for tr in trait_results.values():
        if tr.get("scores") is not None:
            all_scored.update(tr["scores"].keys())
    if not all_scored:
        all_scored = set(sample_ids_all)
    common_samples = sorted(all_scored)

    data = {"sample_id": list(common_samples)}
    for trait_name, tr in trait_results.items():
        sc = tr.get("scores", {})
        data[f"{trait_name}_score"] = [sc.get(s, 0) for s in common_samples]

    df = pd.DataFrame(data)
    score_cols = [c for c in df.columns if c.endswith("_score") and c != "total_score"]
    df["total_score"] = df[score_cols].sum(axis=1).astype(int)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    return df


### Write breeding report: scored rankings, elite lines, complementary cross recommendations.
def write_breeding_report(breeding_df, trait_results, prefix):
    """Writes {prefix}_breeding_report.txt."""
    report_path = f"{prefix}_breeding_report.txt"
    score_cols = [c for c in breeding_df.columns if c.endswith("_score") and c != "total_score"]
    trait_names = [c.replace("_score", "") for c in score_cols]
    n_traits = len(trait_names)

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  BREEDING SELECTION REPORT (Cophenetic Score)\n")
        f.write("=" * 70 + "\n\n")

        # Per-trait summary
        f.write("--- Per-trait summary ---\n")
        for trait_name, tr in trait_results.items():
            thresh = tr["threshold"]
            sc = tr.get("scores", {})
            if sc:
                e1 = tr.get("e1", "?")
                e2 = tr.get("e2", "?")
                e1_score = sc.get(e1, "?")
                e2_score = sc.get(e2, "?")
                pos = sum(1 for v in sc.values() if v > 0)
                neg = sum(1 for v in sc.values() if v < 0)
                zero = sum(1 for v in sc.values() if v == 0)
                f.write(f"  {trait_name}: {e1}(e1)={e1_score:+d}, {e2}(e2)={e2_score:+d}, "
                        f"positive={pos}, negative={neg}, zero={zero}\n")
            else:
                f.write(f"  {trait_name}: no scores\n")
        f.write("\n")

        # Top samples
        f.write("--- Top samples (ranked by total score) ---\n")
        top_n = min(40, len(breeding_df))
        # Build header: Rank, Sample, trait scores..., Total
        col_w = max(7, max(len(t) for t in trait_names) + 1)
        header = f"{'Rank':<6}{'Sample':<14}" + "".join(f"{t:>{col_w}}" for t in trait_names) + f"{'Total':>8}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for rank, (_, row) in enumerate(breeding_df.head(top_n).iterrows(), 1):
            scores = "".join(f"{int(row[c]):>{col_w}}" for c in score_cols)
            f.write(f"{rank:<6}{row['sample_id']:<14}{scores}{int(row['total_score']):>8}\n")
        f.write("\n")

        # Top single samples
        max_total = breeding_df["total_score"].max()
        top_lines = breeding_df[breeding_df["total_score"] == max_total]
        f.write(f"--- Top breeding lines (total_score={int(max_total)}) ---\n")
        for _, row in top_lines.iterrows():
            per_trait = ", ".join(f"{t}={int(row[c]):+d}" for t, c in zip(trait_names, score_cols))
            f.write(f"  {row['sample_id']}  ({per_trait})\n")

        # Complementary crosses: find pairs where each has high scores on different traits
        f.write("\n--- Recommended complementary crosses ---\n")
        best_pair = None
        best_combined = -9999
        top_candidates = breeding_df.head(100)
        for i in range(len(top_candidates)):
            for j in range(i + 1, len(top_candidates)):
                combined = 0
                for c in score_cols:
                    combined += max(top_candidates.iloc[i][c], top_candidates.iloc[j][c])
                if combined > best_combined:
                    best_combined = combined
                    best_pair = (top_candidates.iloc[i], top_candidates.iloc[j])

        if best_pair is not None:
            a, b = best_pair
            a_traits = [f"{t}={int(a[c]):+d}" for t, c in zip(trait_names, score_cols)]
            b_traits = [f"{t}={int(b[c]):+d}" for t, c in zip(trait_names, score_cols)]
            f.write(f"  Best cross pair (combined score={int(best_combined)}):\n")
            f.write(f"    {a['sample_id']} ({'; '.join(a_traits)})\n")
            f.write(f"    {b['sample_id']} ({'; '.join(b_traits)})\n")

        f.write("\n" + "=" * 70 + "\n")
    print(f"[+] Breeding report saved to {report_path}")


### Generate heatmap of per-trait cophenetic scores across samples.
def plot_breeding_heatmap(breeding_df, trait_results, prefix):
    """Generates {prefix}_breeding_matrix.pdf."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    score_cols = [c for c in breeding_df.columns if c.endswith("_score") and c != "total_score"]
    trait_names = [c.replace("_score", "") for c in score_cols]
    n_traits = len(trait_names)

    # Use top 100 samples max to keep the plot readable
    df_plot = breeding_df.head(100)
    n_plot = len(df_plot)

    # Build score matrix
    matrix = np.zeros((n_plot, n_traits))
    avg_scores = np.zeros(n_plot)
    for i, (_, row) in enumerate(df_plot.iterrows()):
        for j, c in enumerate(score_cols):
            matrix[i, j] = row[c]
        avg_scores[i] = row['total_score'] / n_traits

    # Colorblind-safe diverging palette (ColorBrewer RdBu): blue = favorable, red = unfavorable
    cmap = sns.diverging_palette(10, 250, s=85, l=50, as_cmap=True)
    norm = matplotlib.colors.Normalize(vmin=-100, vmax=100)

    fig_height = max(6, n_plot * 0.22)
    fig, ax_heatmap = plt.subplots(figsize=(max(7, n_traits * 1.4), fig_height))
    fig.subplots_adjust(left=0.18, right=0.93, top=0.95, bottom=0.08)

    # ---- Heatmap (draw top-to-bottom, no inversion needed) ----
    for i in range(n_plot):
        y_bottom = n_plot - 1 - i  # row 0 (best) at top
        for j in range(n_traits):
            val = matrix[i, j]
            color = cmap(norm(val))
            ax_heatmap.add_patch(plt.Rectangle((j, y_bottom), 1, 1, facecolor=color, edgecolor='white', lw=0.5))
            text_color = 'white' if abs(val) > 50 else 'black'
            ax_heatmap.text(j + 0.5, y_bottom + 0.5, f"{int(val):+d}", ha='center', va='center',
                            fontsize=7, color=text_color)

    ax_heatmap.set_xlim(0, n_traits)
    ax_heatmap.set_ylim(0, n_plot)
    ax_heatmap.set_xticks([j + 0.5 for j in range(n_traits)])
    ax_heatmap.set_xticklabels(trait_names, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticks([])
    for i in range(n_plot):
        y_center = n_plot - 0.5 - i
        ax_heatmap.text(-0.1, y_center, df_plot['sample_id'].values[i],
                        ha='right', va='center', fontsize=5.5, clip_on=False)
    ax_heatmap.set_title("Breeding Selection Matrix", pad=10)
    ax_heatmap.set_xlabel("Trait")
    ax_heatmap.set_ylabel("Sample (ranked by total score)")

    # ---- Bar chart: use divider for guaranteed alignment ----
    divider = make_axes_locatable(ax_heatmap)
    ax_bar = divider.append_axes("right", size="15%", pad=0.08, sharey=ax_heatmap)

    y_positions = np.array([n_plot - 0.5 - i for i in range(n_plot)])
    bar_colors = ['#2166AC' if v > 0 else '#B2182B' if v < 0 else '#999999' for v in avg_scores]
    ax_bar.barh(y_positions, avg_scores, height=1.0,
                color=bar_colors, edgecolor='none')
    ax_bar.axvline(0, color='#666666', linewidth=0.6)
    ax_bar.set_xlim(-105, 105)
    ax_bar.set_xlabel("Average score")
    ax_bar.set_ylim(0, n_plot)
    ax_bar.set_yticks([])
    ax_bar.set_yticklabels([])

    # ---- Color bar below ----
    cax = divider.append_axes("bottom", size="3%", pad=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label("Cophenetic score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(f"{prefix}_breeding_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Breeding matrix heatmap saved to {prefix}_breeding_matrix.pdf")


### Main pipeline: per-trait haplotype analysis + breeding selection merge.
def bro(traits, prefix, max_reps=None, phenotype_label='phenotype', anchors=None,
        phenotype_file=None):
    """traits: list of (name, vcf, e1, e2). Runs per-trait clustering, then merges into breeding outputs."""
    observed = load_phenotype_file(phenotype_file) if phenotype_file else None
    all_trait_results = {}
    all_sample_ids = set()

    for trait_name, vcf, e1, e2 in traits:
        trait_prefix = f"{prefix}_{trait_name}"
        print(f"\n{'='*50}\n  Processing trait: {trait_name} (vcf={vcf})\n{'='*50}")
        clu_df, n_clu, dist_mat, _, sample_ids, hap_array = uclu(vcf, trait_prefix)
        all_sample_ids.update(sample_ids)
        tr = treebase_hap(dist_mat, sample_ids, [(trait_name, e1, e2)], trait_prefix)
        tr[trait_name]["e1"] = e1
        tr[trait_name]["e2"] = e2
        all_trait_results[trait_name] = tr[trait_name]
        plot_heatmap(dist_mat, sample_ids, filename=f'{trait_prefix}_Pairwise_Hamming_Distanced_Heatmap.pdf')

        base_df = vcf_to_base_matrix(vcf)
        reps = n_clu if max_reps is None else max_reps
        cm_for_plot = tr[trait_name]["cluster_map"]
        if cm_for_plot is not None:
            cm_for_plot = {sid: f"H{cid}" for sid, cid in cm_for_plot.items()}
        else:
            cm_for_plot = {row['sampleID']: f"H{row['cluster']}" for _, row in clu_df.iterrows()}
        plot_pca(hap_array, sample_ids, cm_for_plot, e1, e2, trait_prefix)
        plot_phenotype_boxplot(sample_ids, cm_for_plot, trait_prefix, observed=observed,
                               anchors=anchors, fallback_anchors={e1: 20.0, e2: 60.0},
                               phenotype_label=phenotype_label)
        plot_base_hap_table_from_clustering(base_df, clu_df, cm_for_plot, e1, e2,
                                            output_file=f'{trait_prefix}_hap_base_table.pdf', max_reps=reps)
        plot_tree_with_base_table(f"{trait_prefix}_linkage_matrix.npy",
                                  f"{trait_prefix}_hap_base_table.pdf".replace('.pdf', '_full_table.txt'),
                                  f"{trait_prefix}_tree_with_base_table.pdf",
                                  color_threshold=tr[trait_name]["threshold"])

    # Breeding merge
    print(f"\n{'='*50}\n  Building breeding selection matrix\n{'='*50}")
    breeding_df = build_breeding_matrix(all_sample_ids, all_trait_results)
    write_breeding_report(breeding_df, all_trait_results, prefix)
    plot_breeding_heatmap(breeding_df, all_trait_results, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-trait haplotype clustering and breeding selection from VCF files.")
    parser.add_argument('pop_vcf', type=str, nargs='?', default=None,
                        help='Path to the input VCF file (single-trait / backward compat mode).')
    parser.add_argument('--trait', type=str, action='append', dest='traits', default=[],
                        metavar='name:vcf,e1,e2',
                        help='Define a trait. Format: name:vcf,e1,e2. Repeat for multiple traits.')
    parser.add_argument('--e1acc', type=str, default=None,
                        help='(Backward compat) Favorable accession ID.')
    parser.add_argument('--e2acc', type=str, default=None,
                        help='(Backward compat) Unfavorable accession ID.')
    parser.add_argument('--prefix', type=str, default='result', help='Prefix for all output files.')
    parser.add_argument('--max_reps', type=int, default=None,
                        help='Maximum number of representative haplotypes to plot (default: number of clusters).')
    parser.add_argument('--phenotype', type=str, default='phenotype',
                        help='Phenotype name used for the boxplot (default: "phenotype").')
    parser.add_argument('--anchors', type=str, default='W24:20,CGN22692:60',
                        help='Accessions to annotate on the boxplot (must have values in the '
                             'phenotype file), format "acc:value,acc:value".')
    parser.add_argument('--phenotype-file', type=str, default=None,
                        help='Two-column phenotype file (sample_id, value; tab/space/comma '
                             'separated, header allowed). Required to activate the phenotype '
                             'boxplot for all traits; without it no boxplot is produced.')
    args = parser.parse_args()

    # Normalize to traits list: list of (name, vcf, e1, e2)
    traits = []
    for t_raw in args.traits:
        if ':' not in t_raw:
            parser.error(f"--trait format must be name:vcf,e1,e2, got '{t_raw}'")
        name, rest = t_raw.split(':', 1)
        parts = rest.split(',')
        if len(parts) < 3:
            parser.error(f"--trait format must be name:vcf,e1,e2, got '{t_raw}'")
        vcf, e1, e2 = parts[0], parts[1], parts[2]
        traits.append((name, vcf, e1, e2))

    if not traits and args.pop_vcf is not None:
        print("[i] Using backward-compatible single-trait mode.")
        traits = [("default", args.pop_vcf, args.e1acc, args.e2acc)]

    if not traits:
        parser.error("Either --trait or a VCF file + --e1acc/--e2acc must be provided.")

    anchors = {}
    for item in args.anchors.split(','):
        if ':' not in item:
            parser.error(f"--anchors format must be acc:value,acc:value, got '{item}'")
        acc, val = item.split(':', 1)
        anchors[acc.strip()] = float(val)
    bro(traits, args.prefix, args.max_reps, phenotype_label=args.phenotype, anchors=anchors,
        phenotype_file=args.phenotype_file)


