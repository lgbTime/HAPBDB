# HAPBDB — Haplotype-Aware Phenotype Based Design Breeding

A Python tool for haplotype clustering and genetic distance visualization from VCF (Variant Call Format) files. HAPBDB performs unsupervised hierarchical clustering on samples using Hamming distance over one-hot encoded haplotypes, then generates publication-quality visualizations: dendrograms, pairwise heatmaps, and base-level genotype tables. When extreme-phenotype accessions are specified, it automatically finds the minimum distance threshold that separates them — directly informing breeding selection decisions.

The project includes two versions:

| Version | Script | Use case |
|---------|--------|----------|
| **Single-trait** | `HAPBDB.py` | One VCF, one trait, one pair of extreme accessions |
| **Multi-trait** | `multi_traits_HAPBDB/mutil_traits_HAPBDB.py` | Multiple traits, each with its own VCF and extreme accessions, merged into a breeding selection matrix |

---

## Single-trait HAPBDB

### How It Works

1. **VCF parsing** — Converts each sample's genotype calls (`0/0`, `0/1`, `1/1`) into a flattened one-hot encoding per variant, producing a sample × (variants × 3) matrix. In parallel, genotypes are decoded to nucleotide bases for human-readable base table output.

2. **Hamming distance** — Computes the pairwise Hamming distance matrix across all samples. Samples with similar haplotype profiles have lower distances.

3. **Hierarchical clustering** — Builds a linkage matrix via average-linkage agglomerative clustering and renders a dendrogram.

4. **Threshold search** (when `--e1acc` and `--e2acc` are provided) — Sweeps thresholds from 1.0 down to 0.01 until the two extreme accessions land in different clusters. This threshold is the minimum genetic distance needed to separate favorable from unfavorable phenotypes — a data-driven cutoff for breeding decisions.

5. **Visualizations** — Generates four PDF figures and tabular outputs.

### Output files

| File | Description |
|------|-------------|
| `{prefix}_distance_matrix.npy` | Pairwise Hamming distance matrix (NumPy) |
| `{prefix}_linkage_matrix.npy` | Hierarchical clustering linkage matrix (NumPy) |
| `{prefix}_sample_ids.txt` | Sample IDs in analysis order |
| `{prefix}_strict_all_hap.txt` | Per-sample cluster assignment from unsupervised clustering |
| `{prefix}_tree_base_dis1_haplotype.txt` | Cluster membership at distance threshold = 1.0 |
| `{prefix}_tree_dist{threshold}_extram_based.txt` | Cluster membership at the separation threshold |
| `{prefix}_hap_base_full_table.txt` | Full base-level genotype table (samples × variants) |
| `{prefix}_tree_dendrogram.pdf` | Hierarchical clustering dendrogram |
| `{prefix}_Pairwise_Hamming_Distanced_Heatmap.pdf` | Pairwise distance heatmap |
| `{prefix}_hap_base_table.pdf` | Representative haplotype base table |
| `{prefix}_tree_with_base_table.pdf` | Side-by-side dendrogram and base table |

### Usage

```bash
python HAPBDB.py <vcf_file> [--e1acc ID] [--e2acc ID] [--prefix PREFIX] [--max_reps N]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `pop_vcf` | Yes | — | Path to input VCF file |
| `--e1acc` | No | `None` | Favorable extreme accession ID |
| `--e2acc` | No | `None` | Unfavorable extreme accession ID |
| `--prefix` | No | `result` | Prefix for all output files |
| `--max_reps` | No | *n_clusters* | Max representative haplotypes in base table plot |

**Example:**
```bash
python HAPBDB.py QTL1.vcf --e1acc CGN22050 --e2acc CGN22692 --prefix my_analysis
```

---

## Multi-trait HAPBDB

`multi_traits_HAPBDB/mutil_traits_HAPBDB.py` extends the single-trait pipeline to handle **multiple traits simultaneously**. Each trait has its own VCF file and pair of extreme accessions. After per-trait clustering and threshold detection, it merges all results into a unified breeding selection framework.

### Key additions over single-trait

#### Cophenetic distance scoring

For each trait, every sample receives a score in **[-100, +100]** based on its cophenetic distance to the two extreme accessions:

- **+100** — genetically identical to the favorable extreme (e1)
- **-100** — genetically identical to the unfavorable extreme (e2)
- **0** — equidistant, or beyond the e1–e2 distance

The score quantifies how favorable each sample's haplotype is for each trait.

#### Breeding selection matrix

Scores from all traits are combined into a single DataFrame. Each sample gets:

- Per-trait score columns (`{trait}_score`)
- A `total_score` column (sum of all trait scores)

Samples are ranked by total score — highest first — so breeders can identify lines that perform well across all target traits.

#### Breeding report (`{prefix}_breeding_report.txt`)

A structured text report containing:

1. **Per-trait summary** — extreme accession scores, count of positive/negative/zero samples
2. **Ranked sample table** (top 40) — per-trait scores and total, sorted descending
3. **Top breeding lines** — samples with the maximum total score
4. **Complementary cross recommendations** — pairs of lines that maximize combined trait coverage (one may be strong on trait A while the other is strong on trait B), identified by scanning the top 100 candidates

#### Breeding matrix heatmap (`{prefix}_breeding_matrix.pdf`)

A red-white-green diverging heatmap of the top 100 samples × all traits, with a sidebar bar chart showing each sample's average score. Green = favorable, red = unfavorable.

### Usage

```bash
python mutil_traits_HAPBDB.py \
    --trait t1:QTL_trait1.vcf,e1_acc,e2_acc \
    --trait t2:QTL_trait2.vcf,e1_acc,e2_acc \
    --prefix results/multi
```

Each `--trait` argument has the format `name:vcf,e1,e2` and can be repeated for as many traits as needed.

| Argument | Description |
|----------|-------------|
| `--trait name:vcf,e1,e2` | Define one trait (repeat for each trait) |
| `--prefix` | Output prefix (directory paths allowed) |

**Backward compatible:** You can also call it with a single VCF and `--e1acc`/`--e2acc` like the single-trait version.

### Multi-trait output files

Everything from single-trait (per trait, prefixed `{prefix}_{trait_name}_*`) plus:

| File | Description |
|------|-------------|
| `{prefix}_breeding_report.txt` | Ranked sample scores, top lines, cross recommendations |
| `{prefix}_breeding_matrix.pdf` | Score heatmap across all samples and traits |

### Multi-trait example

```bash
cd multi_traits_HAPBDB
bash multi_traits.sh
```

Runs 3 traits (`t1`, `t2`, `t3`) each from a 100-sample × 80-variant VCF with different extreme accession pairs, generating per-trait visualizations plus the breeding report and heatmap under `demo_data_out/`.

---

## Requirements

- Python 3.6+
- numpy ≥ 1.18, pandas ≥ 1.0, scikit-learn ≥ 0.22, scipy ≥ 1.4
- matplotlib ≥ 3.1, seaborn ≥ 0.10
- pycairo ≥ 1.19 (optional; falls back to matplotlib's PDF backend)

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
HAPBDB/
├── HAPBDB.py                      # Single-trait analysis pipeline
├── run.sh                         # Entry point → runs example.sh
├── example.sh                     # Single-trait demo
├── requirements.txt               # Python dependencies
├── demo_data/
│   ├── extract_demo_vcf.py        # Extracts demo VCF from QTL1.vcf
│   └── demo.vcf                   # 80 samples × 80 variants (includes CGN22050 & CGN22692)
├── multi_traits_HAPBDB/
│   ├── mutil_traits_HAPBDB.py     # Multi-trait pipeline with breeding scores
│   ├── multi_traits.sh            # Multi-trait demo (3 traits)
│   ├── demo_data/
│   │   ├── QTL_trait1_100.vcf     # 100 samples × 80 variants
│   │   ├── QTL_trait2_100.vcf
│   │   └── QTL_trait3_100.vcf
│   └── demo_data_out/             # Multi-trait demo outputs
└── QTL1.vcf / QTL2.vcf            # Full source VCFs
```

## Use Case

HAPBDB is designed for **marker-assisted breeding programs**:

- Given QTL-associated VCF files and accessions at opposite trait extremes, identify the genetic distance threshold that separates favorable from unfavorable haplotypes.
- Base table visualizations show which specific SNP alleles distinguish the clusters, helping breeders choose crosses that break undesirable linkages.
- The multi-trait version ranks breeding lines across multiple target traits simultaneously and recommends complementary cross pairs.

## License

MIT License.

## Citation

If you use HAPBDB in your research, please cite the repository.
