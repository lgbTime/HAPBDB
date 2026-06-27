#!/usr/bin/env python3
"""
Extract a demo VCF from QTL1.vcf.
Selects ~80 samples (always including CGN22695 and CGN22692)
and uses all 80 variants from the source.
"""
import random

random.seed(42)

source_vcf = "../QTL1.vcf"
output_vcf = "demo.vcf"

# --- Read the source VCF ---
header_lines = []
sample_ids = []
variant_lines = []

with open(source_vcf) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("##"):
            header_lines.append(line)
        elif line.startswith("#CHROM"):
            parts = line.split("\t")
            sample_ids = parts[9:]
            header_lines.append(line)
        else:
            variant_lines.append(line)

print(f"Source VCF: {len(sample_ids)} samples, {len(variant_lines)} variants")

# --- Select samples ---
must_include = ["CGN22050", "CGN22692"]
target_n = 80

# Verify must_include exist
for sid in must_include:
    if sid not in sample_ids:
        raise SystemExit(f"ERROR: {sid} not found in VCF")

remaining = [s for s in sample_ids if s not in must_include]
n_pick = target_n - len(must_include)
picked = random.sample(remaining, min(n_pick, len(remaining)))
selected = must_include + picked

# Get column indices in original order
sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
selected_indices = [sample_to_idx[s] for s in selected]

print(f"Selected: {len(selected)} samples (includes {must_include})")
print(f"Variants:  {len(variant_lines)} (all retained)")

# --- Write output ---
with open(output_vcf, "w") as out:
    for h in header_lines:
        if h.startswith("#CHROM"):
            fields = h.split("\t")
            new_header = "\t".join(fields[:9] + selected)
            out.write(new_header + "\n")
        else:
            out.write(h + "\n")
    for vline in variant_lines:
        fields = vline.split("\t")
        new_fields = fields[:9] + [fields[9 + idx] for idx in selected_indices]
        out.write("\t".join(new_fields) + "\n")

print(f"Wrote {output_vcf}")
