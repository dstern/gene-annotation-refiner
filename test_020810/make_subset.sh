#!/bin/bash
# Extract scaffold_2:77870000-77990000 subset for gene 020810 trace.
# Gene coords: 77920227-77935851 (~16 kb).  100 kb flanking window.
set -euo pipefail

SEQID="scaffold_2"
REG_START=77870000
REG_END=77990000

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/test_run"
OUT="$(cd "$(dirname "$0")" && pwd)"

GENOME="$SRC/Acyrthosiphon_pisum_JIC1_v1.0.scaffolds_mtDNA.fa"
HELIXER="$SRC/Apis_helixer_3v25.sorted.gff"
STRINGTIE="$SRC/Apis.mixed.f0.75.gtf"
TRANSDECODER="$SRC/Acyrthosiphon_pisum.27xi23_isoseq.transdecoder_isoseqUpdated.updated.12xii23.gff3"
BIGWIG="$SRC/Apis.bam.bw"
JUNCTIONS="$SRC/portcullis_filtered.pass.junctions.tab"

echo "Region: ${SEQID}:${REG_START}-${REG_END}"

awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$HELIXER" > "$OUT/subset_helixer.gff"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$STRINGTIE" > "$OUT/subset_stringtie.gtf"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$TRANSDECODER" > "$OUT/subset_transdecoder.gff3"
head -1 "$JUNCTIONS" > "$OUT/subset_junctions.tab"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    'NR>1 && $3==s && $5+1>=a && $6<=b' "$JUNCTIONS" >> "$OUT/subset_junctions.tab"

echo "GFF subsets written."
echo "(Using full genome + full bigwig from test_run/ — no need to subset those.)"
ls -lh "$OUT"/subset_*
