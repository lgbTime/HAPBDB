#!/data/lgb/software/anaconda3/bin/python
import matplotlib
matplotlib.use('Agg')
# --------------------------------
# Adobe‑editable fonts
# --------------------------------
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import argparse
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.cluster.hierarchy as sch
import os
try:
    import cairo
    CAIRO_BACKEND = "cairo"
except ImportError:
    CAIRO_BACKEND = "pdf"

### Construct haplotype tree and perform clustering analysis based on hierarchical clustering
def treebase_hap(distance_matrix, sample_ids, e1, e2, prefix):
    hpix = int(len(sample_ids) / 12)
    sample_ids = np.array(sample_ids)
    linkage_matrix = sch.linkage(distance_matrix, method='average')
    np.save(f"{prefix}_linkage_matrix.npy", linkage_matrix)
    print(f"linkage_matrix saved to {prefix}_linkage_matrix.npy")
    with open(f"{prefix}_sample_ids.txt", "w") as fout:
        for sid in sample_ids:
            fout.write(sid + "\n")
    plt.figure(figsize=(10, hpix))
    sch.dendrogram(linkage_matrix, labels=sample_ids, orientation='left', leaf_rotation=360)
    plt.title('Hierarchical Clustering Dendrogram')
    with PdfPages(f"{prefix}_tree_dendrogram.pdf") as pdf_pages:
        pdf_pages.savefig()

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
    # Skip dynamic threshold separation if e1 and e2 are not provided
    if e1 is None or e2 is None:
        print("[i] e1acc and/or e2acc not provided, skip dynamic threshold separation.")
        return tree_cluster_map

    # Validate e1 and e2 exist in sample_ids before proceeding
    if e1 not in sample_ids or e2 not in sample_ids:
        print(f"Error: {e1} or {e2} are not in the provided sample_ids.")
        return tree_cluster_map

    for threshold_distance in np.arange(1.0, 0.01, -0.01):
        found = False
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
                return {sid: clusters[i] for i, sid in enumerate(sample_ids)}

    return tree_cluster_map

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
    w, h = int(len(sample_ids)/5), int(len(sample_ids)/5)
    plt.figure(figsize=(w, h))
    sns.heatmap(distance_matrix, xticklabels=sample_ids, yticklabels=sample_ids, cmap='coolwarm', cbar_kws={'label': 'Hamming Distance'})
    plt.title('Pairwise Hamming Distance Heatmap')
    pdf_pages = PdfPages(filename)
    pdf_pages.savefig()
    pdf_pages.close()

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
                        gts.append('/'.join(alleles))
                    except:
                        gts.append('NA')
            base_matrix.append(gts)
    base_df = pd.DataFrame(base_matrix, index=variant_ids, columns=sample_ids).T
    return base_df

### Generate and save a haplotype base table visualization with clustering information.
def plot_base_hap_table_from_clustering(base_df, sample_clusters, acc1, acc2, output_file, max_reps=None, prefix='result'):
    base_colors = {
        'A': '#4daf4a', 'T': '#e41a1c', 'G': '#377eb8', 'C': '#ff7f00', 'NA': '#bdbdbd'
    }
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
    fig, ax = plt.subplots(figsize=(max(8, len(filtered_df.columns)*0.4), max(4, len(filtered_df)*0.5)))
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
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, edgecolor='black'))
            ax.text(j+0.5, i+0.5, display, ha='center', va='center', fontsize=10)
    ax.set_xticks([i + 0.5 for i in range(len(filtered_df.columns))])
    ax.set_xticklabels(filtered_df.columns, rotation=45, fontsize=8, ha='right')
    ax.set_yticks([i + 0.5 for i in range(len(filtered_df.index))])
    ax.set_yticklabels(filtered_df.index, fontsize=8)
    ax.set_xlim(0, len(filtered_df.columns))
    ax.set_ylim(0, len(filtered_df.index))
    ax.invert_yaxis()
    ax.set_title("Haplotype Base Table", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='pdf', bbox_inches='tight', backend=CAIRO_BACKEND)
    print(f"[+] Haplotype base plot saved to: {output_file}")

### Generate and save a combined visualization of dendrogram and haplotype base table.
def plot_tree_with_base_table(linkage_matrix_file, hap_base_file, output_pdf, prefix="result"):
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
    dendro = sch.dendrogram(linkage_matrix, orientation='left', labels=base_df.index, ax=ax1)
    leaf_order = dendro['ivl']   # Get leaf order from dendrogram
    leaf_order_for_table = leaf_order[::-1]

    # --- Right panel: base table ---
    ordered_df = base_df.reindex(leaf_order_for_table)
    ax2 = plt.subplot(gs[1])

    base_colors = {
        'A': '#4daf4a', 'T': '#e41a1c', 'G': '#377eb8',
        'C': '#ff7f00', 'NA': '#bdbdbd'
    }

    for i, sample in enumerate(ordered_df.index):
        for j, val in enumerate(ordered_df.columns):
            base = ordered_df.loc[sample, val]
            if base == 'NA':
                color, text = base_colors['NA'], 'NA'
            else:
                bases = str(base).split('/')
                text = ''.join(sorted(set(bases)))
                color = base_colors.get(bases[0], '#ffffff')
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black'))
            ax2.text(j+0.5, i+0.5, text, ha='center', va='center', fontsize=6)

    ax2.set_xlim(0, len(ordered_df.columns))
    ax2.set_ylim(0, len(ordered_df.index))
    ax2.invert_yaxis()
    ax2.set_xticks(np.arange(len(ordered_df.columns)) + 0.5)
    ax2.set_xticklabels(ordered_df.columns, rotation=90, fontsize=6)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[+] Combined tree + haplotype base table saved to {output_pdf}")

### Main pipeline function for haplotype clustering and visualization.
def bro(pop_vcf, acc1, acc2, prefix, max_reps=None):
    clu_df, n_clu, distance_matrix_all, unique_indices, sample_ids, haplotypes_array = uclu(pop_vcf, prefix)
    tree_cluster_map = treebase_hap(distance_matrix_all, sample_ids, acc1, acc2, prefix)
    plot_heatmap(distance_matrix_all, sample_ids, filename=f'{prefix}_Pairwise_Hamming_Distanced_Heatmap.pdf')
    base_df = vcf_to_base_matrix(pop_vcf)
    reps_to_use = n_clu if max_reps is None else max_reps
    plot_base_hap_table_from_clustering(base_df, clu_df, acc1, acc2, output_file=f'{prefix}_hap_base_table.pdf', max_reps=reps_to_use, prefix=prefix)
    # Generate combined tree + base table plot
    linkage_matrix_file = f"{prefix}_linkage_matrix.npy"
    hap_base_file = f"{prefix}_hap_base_full_table.txt"
    output_pdf = f"{prefix}_tree_with_base_table.pdf"
    plot_tree_with_base_table(linkage_matrix_file, hap_base_file, output_pdf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster haplotypes from VCF files and identify tree-based distance thresholds separating two extreme-phenotype accessions.")
    parser.add_argument('pop_vcf', type=str, help='Path to the input VCF file.')
    parser.add_argument('--e1acc', type=str, default=None, help='The extreme phenotype accession that has a favorable trait (ID must be in the VCF).')
    parser.add_argument('--e2acc', type=str, default=None, help='The extreme phenotype accession that has an unfavorable trait (ID must be in the VCF).')
    parser.add_argument('--prefix', type=str, default='result', help='Prefix for all output files.')
    parser.add_argument('--max_reps', type=int, default=None, help='Maximum number of representative haplotypes to plot (default: number of clusters).')
    args = parser.parse_args()
    bro(args.pop_vcf, args.e1acc, args.e2acc, args.prefix, args.max_reps)


