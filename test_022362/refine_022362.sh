#!/bin/bash
# Test run on the scaffold_2:76722198-77114949 subset
# (100kb flanking gene Apis_refined_14042026_022362)
#
# Run from the repo root or the test_022362 directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/../gene_annotation_refiner_2.py" \
    --genome   "$SCRIPT_DIR/../test_run/Acyrthosiphon_pisum_JIC1_v1.0.scaffolds_mtDNA.fa" \
    --output   "$SCRIPT_DIR/refine_022362_output.gff" \
    --helixer  "$SCRIPT_DIR/subset_helixer.gff" \
    --stringtie "$SCRIPT_DIR/subset_stringtie.gtf" \
    --transdecoder "$SCRIPT_DIR/subset_transdecoder.gff3" \
    --bigwig   "$SCRIPT_DIR/subset_coverage.bw" \
    --junctions "$SCRIPT_DIR/subset_junctions.tab" \
    --renumber \
    --gene_prefix Apis_022362_test \
    2>&1 | tee "$SCRIPT_DIR/refine_022362.log"
