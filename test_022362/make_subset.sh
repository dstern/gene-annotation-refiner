#!/bin/bash
# Run this on the HPC to extract scaffold_2:76722198-77114949
# (100kb flanking gene Apis_refined_14042026_022362) from all input files.
#
# Usage: bash make_subset.sh
# Output files written to the directory containing this script.

set -euo pipefail

SEQID="scaffold_2"
REG_START=76722198
REG_END=77114949

Acyrthosiphon_pisum_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum'
JICv1_0_mapped_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum/Annotate/JICv1.0_mapped'
Apis_annotate_4May2025_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum/Annotate/JICv1.0_mapped/Apis_annotate_4May2025'

GENOME="$Acyrthosiphon_pisum_folder/Genomes/v1.0_JIC/Acyrthosiphon_pisum_JIC1_v1.0.scaffolds_mtDNA.fa"
HELIXER="$Apis_annotate_4May2025_folder/Apis_helixer_3v25.sorted.gff"
STRINGTIE="$Apis_annotate_4May2025_folder/Apis.mixed.f0.75.gtf"
TRANSDECODER="$JICv1_0_mapped_folder/Acyrthosiphon_pisum.27xi23_isoseq.transdecoder_isoseqUpdated.updated.12xii23.gff3"
BIGWIG="$JICv1_0_mapped_folder/Apis.bam.bw"
JUNCTIONS="$Apis_annotate_4May2025_folder/portcullis_out_all/3-filt/portcullis_filtered.pass.junctions.tab"

OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Subsetting to ${SEQID}:${REG_START}-${REG_END}"
echo "Output dir: $OUT_DIR"

ml conda
conda activate refine_annot_env

# ── Genome FASTA ──────────────────────────────────────────────────────────────
echo "  genome..."
samtools faidx "$GENOME" "${SEQID}:${REG_START}-${REG_END}" \
    | sed "s/>${SEQID}:${REG_START}-${REG_END}/>$SEQID/" \
    > "$OUT_DIR/subset_genome.fa"
samtools faidx "$OUT_DIR/subset_genome.fa"

# ── Helixer GFF ───────────────────────────────────────────────────────────────
echo "  helixer..."
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$HELIXER" \
    > "$OUT_DIR/subset_helixer.gff"

# ── StringTie GTF ─────────────────────────────────────────────────────────────
echo "  stringtie..."
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$STRINGTIE" \
    > "$OUT_DIR/subset_stringtie.gtf"

# ── TransDecoder GFF3 ─────────────────────────────────────────────────────────
echo "  transdecoder..."
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$TRANSDECODER" \
    > "$OUT_DIR/subset_transdecoder.gff3"

# ── BigWig (subset region, rebase coordinates to 1-based within region) ───────
# Note: the subset bigwig keeps the original scaffold_2 coordinates so that
# the pipeline does not need to be modified.  We write the full scaffold_2
# track header but only populate the region of interest; all other positions
# will return NaN (treated as 0 by the pipeline).
echo "  bigwig..."
python3 - "$BIGWIG" "$OUT_DIR/subset_coverage.bw" \
          "$SEQID" "$REG_START" "$REG_END" <<'PYEOF'
import pyBigWig, sys, numpy as np
src, dst, seqid = sys.argv[1], sys.argv[2], sys.argv[3]
reg_start, reg_end = int(sys.argv[4]), int(sys.argv[5])

bw_in = pyBigWig.open(src)
# Get original chromosome size so coordinate space is preserved
orig_size = bw_in.chroms()[seqid]

bw_out = pyBigWig.open(dst, 'w')
bw_out.addHeader([(seqid, orig_size)])

# pyBigWig: 0-based half-open
vals = bw_in.values(seqid, reg_start - 1, reg_end, numpy=True)
vals = np.nan_to_num(vals.astype(float), nan=0.0)

# Write only non-zero spans to keep file small
starts, ends, values = [], [], []
i = 0
while i < len(vals):
    if vals[i] > 0:
        j = i
        while j < len(vals) and vals[j] > 0:
            j += 1
        starts.append(reg_start - 1 + i)
        ends.append(reg_start - 1 + j)
        values.append(float(np.mean(vals[i:j])))
        # Write per-base for accuracy
        for k in range(i, j):
            bw_out.addEntries([seqid], [reg_start - 1 + k],
                              ends=[reg_start + k], values=[float(vals[k])])
        i = j
    else:
        i += 1

bw_in.close()
bw_out.close()
print(f"  BigWig written: {dst}")
PYEOF

# ── Junctions TAB ─────────────────────────────────────────────────────────────
echo "  junctions..."
head -1 "$JUNCTIONS" > "$OUT_DIR/subset_junctions.tab"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    'NR>1 && $3==s && $5+1>=a && $6<=b' "$JUNCTIONS" \
    >> "$OUT_DIR/subset_junctions.tab"

echo ""
echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"/subset_*

conda deactivate
