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
def treebase_hap(distance_matrix, sample_ids, e1, e2, prefix):
    """Returns (tree_cluster_map, separation_threshold)."""
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

    tree_cluster_map = {}
    # Perform clustering with default threshold of 1
    with open(f"{prefix}_tree_base_dis1_haplotype.txt", 'w') as tree_dist1:
        threshold_distance = 1
        clusters = sch.fcluster(linkage_matrix, threshold_distance, criterion='distance')
        for cluster_id in np.unique(clusters):
            cluster_samples = sample_ids[clusters == cluster_id]
            tree_dist1.write(f"haplotype_by_tree_distance_{threshold_distance}:\thaplotype:{cluster_id}:\t{','.join(cluster_samples)}\n")
            for sid in cluster_samples:
                tree_cluster_map[sid] = cluster_id

    # Dynamic threshold separation: minimum distance at which e1 and e2 fall into different clusters
    separation_threshold = None
    if e1 is None or e2 is None:
        print("[i] e1acc and/or e2acc not provided, skip dynamic threshold separation.")
    elif e1 not in sample_ids or e2 not in sample_ids:
        print(f"Error: {e1} or {e2} are not in the provided sample_ids.")
    else:
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
                    print(f"At threshold {round(threshold_distance,2)}, {e1} and {e2} are in different clusters.")
                    with open(f"{prefix}_tree_dist{round(threshold_distance,2)}_extram_based.txt", 'w') as tree_base_out_file:
                        for cluster_id in np.unique(clusters):
                            cluster_samples = sample_ids[clusters == cluster_id]
                            tree_base_out_file.write(f"H{cluster_id}:\t{','.join(cluster_samples)}\n")
                    separation_threshold = threshold_distance
                    tree_cluster_map = {sid: clusters[i] for i, sid in enumerate(sample_ids)}
                    break
            if separation_threshold is not None:
                break

    # Dendrogram with branches colored by cluster: below the separation threshold each
    # cluster gets its own color, matching the *_extram_based.txt cluster membership file.
    sch.set_link_color_palette(LINK_COLORS)
    fig, ax = plt.subplots(figsize=(10, hpix))
    sch.dendrogram(linkage_matrix, labels=sample_ids, orientation='left', leaf_rotation=360,
                   leaf_font_size=6, color_threshold=separation_threshold,
                   above_threshold_color=ABOVE_THRESHOLD_COLOR,
                   link_color_func=_link_color_func(linkage_matrix, len(sample_ids), separation_threshold),
                   ax=ax)
    ax.set_xlabel('Genetic distance (Hamming)')
    ax.set_ylabel('Sample')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    with PdfPages(f"{prefix}_tree_dendrogram.pdf") as pdf_pages:
        pdf_pages.savefig()
    plt.close(fig)

    return tree_cluster_map, separation_threshold

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


### Draw a per-haplotype-cluster boxplot of a phenotype: observed values from a
### phenotype file when provided, otherwise simulated from haplotype similarity.
def plot_phenotype_boxplot(haplotypes_array, sample_ids, cluster_map, prefix,
                           anchors=None, fallback_anchors=None, seed=42,
                           phenotype_label='Days to flowering', observed=None):
    """`anchors`: {sample_id: phenotype}, e.g. {'W24': 20.0, 'CGN22692': 60.0}.
    `observed`: {sample_id: float} phenotypes loaded from --phenotype-file.
    Without `observed`, phenotypes are simulated: samples genetically close to an
    anchor get phenotypes close to that anchor (linear interpolation of pairwise
    Hamming distance, plus small noise); `fallback_anchors` are tried when an
    anchor is missing from the data.
    Writes {prefix}_{phenotype}_boxplot.pdf plus {prefix}_{phenotype}_phenotype.txt
    (observed) or {prefix}_{phenotype}_simulated_phenotype.txt (simulated)."""
    slug = ''.join(c if c.isalnum() else '_' for c in phenotype_label.lower()).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    labels = [str(cluster_map.get(sid, 'H?')) for sid in sample_ids] if cluster_map else ['H1'] * len(sample_ids)

    if observed is not None:
        # ----- observed phenotypes -----
        phenos = np.array([observed.get(sid, np.nan) for sid in sample_ids], dtype=float)
        n_obs = int(np.isfinite(phenos).sum())
        if n_obs < 3:
            print(f"[!] phenotype boxplot skipped: only {n_obs} samples with observed phenotypes")
            return
        # annotate anchors that have observed values; else annotate min/max samples
        present = {}
        for acc in (anchors or {}):
            if acc in sample_ids and np.isfinite(phenos[sample_ids.index(acc)]):
                present[acc] = phenos[sample_ids.index(acc)]
        for acc in (fallback_anchors or {}):
            if acc in sample_ids and np.isfinite(phenos[sample_ids.index(acc)]) and acc not in present:
                present[acc] = phenos[sample_ids.index(acc)]
        if not present:
            valid = [(sid, p) for sid, p in zip(sample_ids, phenos) if np.isfinite(p)]
            if valid:
                lo = min(valid, key=lambda t: t[1])
                hi = max(valid, key=lambda t: t[1])
                present = {lo[0]: lo[1], hi[0]: hi[1]}
        source_note = f"observed phenotypes (n={n_obs})"
        out_table = f"{prefix}_{slug}_phenotype.txt"
    else:
        # ----- simulated phenotypes -----
        if anchors is None:
            anchors = {'W24': 20.0, 'CGN22692': 60.0}
        present = {acc: val for acc, val in anchors.items() if acc in sample_ids}
        if len(present) < 2 and fallback_anchors:
            for acc, val in fallback_anchors.items():
                if acc in sample_ids and acc not in present:
                    present[acc] = val
        if len(present) < 2:
            print(f"[!] phenotype boxplot skipped: need >=2 anchor accessions present in the data "
                  f"(anchors={list(anchors)}, found={list(present)})")
            return
        rng = np.random.default_rng(seed)
        hap = np.asarray(haplotypes_array)
        phenos = np.full(len(sample_ids), np.nan)
        acc_items = list(present.items())
        (acc1, ph1), (acc2, ph2) = acc_items[0], acc_items[1]
        d1 = np.mean(hap != hap[sample_ids.index(acc1)], axis=1)
        d2 = np.mean(hap != hap[sample_ids.index(acc2)], axis=1)
        # s in [0,1]: 0 -> like acc1 (e.g. early flowering), 1 -> like acc2 (late)
        s = d1 / (d1 + d2 + 1e-12)
        phenos = ph1 + s * (ph2 - ph1) + rng.normal(0, 1.5, len(sample_ids))
        phenos = np.clip(phenos, min(ph1, ph2), max(ph1, ph2))
        for acc, val in present.items():
            phenos[sample_ids.index(acc)] = val
        source_note = f"simulated from anchors {acc1}={ph1:.0f}, {acc2}={ph2:.0f}"
        out_table = f"{prefix}_{slug}_simulated_phenotype.txt"

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
    # Annotate anchor accessions
    for acc, val in present.items():
        lab = str(cluster_map.get(acc, 'H?'))
        if lab in order:
            x = order.index(lab) + 1
            ax.annotate(f'{acc}\n({val:.0f} d)', (x, val), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=7, fontweight='bold')

    ax.set_ylabel(phenotype_label)
    ax.set_xlabel('Haplotype cluster')
    ax.set_title(f'{phenotype_label} by Haplotype Cluster ({"observed" if observed is not None else "simulated"})', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig(f"{prefix}_{slug}_boxplot.pdf", dpi=300, bbox_inches='tight', format='pdf',
                backend=CAIRO_BACKEND)
    plt.close(fig)

    with open(out_table, "w") as f:
        f.write("sample_id\tcluster\tphenotype\n")
        for sid, lab, p in zip(sample_ids, labels, phenos):
            if np.isfinite(p):
                f.write(f"{sid}\t{lab}\t{p:.2f}\n")
    print(f"[+] {phenotype_label} boxplot saved to {prefix}_{slug}_boxplot.pdf ({source_note})")

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
def plot_base_hap_table_from_clustering(base_df, sample_clusters, acc1, acc2, output_file, max_reps=None, prefix='result'):
    # Colorblind-safe (Okabe-Ito) allele colors, suitable for high-impact journals.
    # 'I' = indel, 'SV' = structural variant.
    base_colors = {
        'A': '#0072B2', 'T': '#D55E00', 'G': '#009E73', 'C': '#CC79A7',
        'I': '#E69F00', 'SV': '#8C564B', 'NA': '#BDBDBD'
    }
    legend_alleles = ['A', 'T', 'G', 'C', 'I', 'SV', 'NA']
    cluster_file = [f for f in os.listdir('.') if f.startswith(f"{prefix}_tree_dist") and f.endswith("_extram_based.txt")]
    tree_cluster_map = {}
    if cluster_file:
        with open(cluster_file[0]) as f:
            for line in f:
                parts = line.strip().split('\t')
                cluster_id = parts[0].rstrip(':')
                samples = parts[1].split(',')
                for s in samples:
                    tree_cluster_map[s] = cluster_id
    else:
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
    base_df.to_csv(f"{prefix}_hap_base_full_table.txt", sep='\t')
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

### Main pipeline function for haplotype clustering and visualization.
def bro(pop_vcf, acc1, acc2, prefix, max_reps=None, phenotype_label='Days to flowering',
        anchors=None, phenotype_file=None):
    clu_df, n_clu, distance_matrix_all, unique_indices, sample_ids, haplotypes_array = uclu(pop_vcf, prefix)
    tree_cluster_map, sep_threshold = treebase_hap(distance_matrix_all, sample_ids, acc1, acc2, prefix)
    plot_pca(haplotypes_array, sample_ids, tree_cluster_map, acc1, acc2, prefix)
    observed = load_phenotype_file(phenotype_file) if phenotype_file else None
    plot_phenotype_boxplot(haplotypes_array, sample_ids, tree_cluster_map, prefix,
                           anchors=anchors, observed=observed, phenotype_label=phenotype_label)
    plot_heatmap(distance_matrix_all, sample_ids, filename=f'{prefix}_Pairwise_Hamming_Distanced_Heatmap.pdf')
    base_df = vcf_to_base_matrix(pop_vcf)
    reps_to_use = n_clu if max_reps is None else max_reps
    plot_base_hap_table_from_clustering(base_df, clu_df, acc1, acc2, output_file=f'{prefix}_hap_base_table.pdf', max_reps=reps_to_use, prefix=prefix)
    # Generate combined tree + base table plot
    linkage_matrix_file = f"{prefix}_linkage_matrix.npy"
    hap_base_file = f"{prefix}_hap_base_full_table.txt"
    output_pdf = f"{prefix}_tree_with_base_table.pdf"
    plot_tree_with_base_table(linkage_matrix_file, hap_base_file, output_pdf, color_threshold=sep_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster haplotypes from VCF files and identify tree-based distance thresholds separating two extreme-phenotype accessions.")
    parser.add_argument('pop_vcf', type=str, help='Path to the input VCF file.')
    parser.add_argument('--e1acc', type=str, default=None, help='The extreme phenotype accession that has a favorable trait (ID must be in the VCF).')
    parser.add_argument('--e2acc', type=str, default=None, help='The extreme phenotype accession that has an unfavorable trait (ID must be in the VCF).')
    parser.add_argument('--prefix', type=str, default='result', help='Prefix for all output files.')
    parser.add_argument('--max_reps', type=int, default=None, help='Maximum number of representative haplotypes to plot (default: number of clusters).')
    parser.add_argument('--phenotype', type=str, default='Days to flowering',
                        help='Phenotype name used for the simulated boxplot (default: "Days to flowering").')
    parser.add_argument('--anchors', type=str, default='W24:20,CGN22692:60',
                        help='Anchor accessions and their phenotype values for simulation, '
                             'format "acc:value,acc:value" (default: "W24:20,CGN22692:60").')
    parser.add_argument('--phenotype-file', type=str, default=None,
                        help='Optional two-column phenotype file (sample_id, value; tab/space/comma '
                             'separated, header allowed). When provided, the boxplot uses these '
                             'observed values instead of simulating phenotypes.')
    args = parser.parse_args()
    anchors = {}
    for item in args.anchors.split(','):
        if ':' not in item:
            parser.error(f"--anchors format must be acc:value,acc:value, got '{item}'")
        acc, val = item.split(':', 1)
        anchors[acc.strip()] = float(val)
    bro(args.pop_vcf, args.e1acc, args.e2acc, args.prefix, args.max_reps,
        phenotype_label=args.phenotype, anchors=anchors, phenotype_file=args.phenotype_file)


