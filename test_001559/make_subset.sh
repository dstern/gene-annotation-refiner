#!/bin/bash
# Subset for scaffold_2:78750000-78920000 (100 kb flanking gene region 78814805-78853517)
set -euo pipefail

SEQID="scaffold_2"
REG_START=78750000
REG_END=78920000

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/test_run"
OUT="$(cd "$(dirname "$0")" && pwd)"

HELIXER="$SRC/Apis_helixer_3v25.sorted.gff"
STRINGTIE="$SRC/Apis.mixed.f0.75.gtf"
TRANSDECODER="$SRC/Acyrthosiphon_pisum.27xi23_isoseq.transdecoder_isoseqUpdated.updated.12xii23.gff3"
JUNCTIONS="$SRC/portcullis_filtered.pass.junctions.tab"

awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$HELIXER" > "$OUT/subset_helixer.gff"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$STRINGTIE" > "$OUT/subset_stringtie.gtf"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    '$1==s && $4>=a && $5<=b' "$TRANSDECODER" > "$OUT/subset_transdecoder.gff3"
head -1 "$JUNCTIONS" > "$OUT/subset_junctions.tab"
awk -v s="$SEQID" -v a="$REG_START" -v b="$REG_END" \
    'NR>1 && $3==s && $5+1>=a && $6<=b' "$JUNCTIONS" >> "$OUT/subset_junctions.tab"

ls -lh "$OUT"/subset_*
