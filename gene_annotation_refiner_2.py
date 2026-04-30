#!/usr/bin/env python3
"""
Gene Annotation Refiner
=======================
Integrates multiple evidence sources (Helixer ML predictions, StringTie RNA-seq
transcripts, TransDecoder CDS predictions, RNA-seq coverage and junctions, and 
human-curated models) to produce a refined GFF file with scores (0-1) for all features.

Mode 1 — Full consensus (default):
    python gene_annotation_refiner.py \
        --genome genome.fa \
        --helixer helixer.gff \
        --stringtie stringtie.gtf \
        --transdecoder transdecoder.gff \
        --bigwig rnaseq.bw \
        --output refined_annotation.gff \
        [--manual_annotation manual.gff] \
        [--bam rnaseq.bam | --junctions junctions.tab]

Mode 2 — Replace an existing annotation with manual annotations:
    python gene_annotation_refiner.py \
        --genome genome.fa \
        --refine_existing previous.gff \
        --manual_annotation manual.gff \
        --output refined_annotation.gff
    Trusts the existing annotation; manual genes replace overlapping ones.
    No evidence-based refinement is run.

Mode 2b — Same as Mode 2, plus RNA-seq evidence-based refinement:
    python gene_annotation_refiner.py \
        --genome genome.fa \
        --refine_existing previous.gff \
        --refine_with_evidence \
        --bigwig rnaseq.bw \
        --output refined_annotation.gff \
        [--manual_annotation manual.gff] \
        [--bam rnaseq.bam | --junctions junctions.tab]

Splice junction evidence:
    The pipeline uses splice junction read counts to validate introns.
    You can provide this evidence in two ways:

    --bam rnaseq.bam       Scan a BAM file for spliced reads (slow — re-scans
                            reads for every intron in every gene).

    --junctions FILE       Use a pre-computed junction file (fast — loaded once
                            into memory at startup, O(1) lookups). Accepts:
                            - Portcullis filtered output (.tab)
                            - STAR splice junction file (SJ.out.tab)
                            - BED format with read counts in score column
                            To generate with Portcullis:
                              portcullis full --threads 8 genome.fa rnaseq.bam
                              # then use: portcullis_out/3-filt/portcullis_filtered.pass.junctions.tab

    If both --bam and --junctions are given, --junctions is used.
    Coverage is always read from --bigwig (not the BAM).

    --bigwig, --bigwig_fwd, --bigwig_rev, --bam, and --junctions all
    accept multiple files. For bigwigs, per-base values are summed
    across files; for junctions, junctions are merged and read counts
    summed across files; for BAMs, queries iterate over all files and
    counts are summed. Example:
        --bigwig sample1.bw sample2.bw sample3.bw
        --junctions sample1.tab sample2.tab

Stranded coverage (optional, recommended when stranded libraries are
available):
    --bigwig is the primary unstranded coverage track (built from all
    libraries). To additionally veto antisense leakage from neighboring
    genes on the opposite strand, supply same-strand bigwigs built only
    from stranded libraries:

      --bigwig_fwd FILE   coverage of transcripts on the + strand
      --bigwig_rev FILE   coverage of transcripts on the - strand

    Naming convention: these flags expect bigwigs labeled by TRANSCRIPT
    strand, not read strand. Under dUTP-protocol libraries (TruSeq
    Stranded, NEB Ultra II Directional, etc.) reads map to the genomic
    strand OPPOSITE the originating transcript. Pipelines that split
    bigwigs by read strand therefore produce files where:

      *forward.bw  (reads on + genomic strand) = − strand transcripts
                                                  -> pass to --bigwig_rev
      *reverse.bw  (reads on − genomic strand) = + strand transcripts
                                                  -> pass to --bigwig_fwd

    deepTools handles the dUTP flip internally, so its outputs are
    transcript-strand-labeled and pass through directly:
      bamCoverage -b stranded.bam --filterRNAstrand forward -o plus_tx.bw
      bamCoverage -b stranded.bam --filterRNAstrand reverse -o minus_tx.bw
    -> --bigwig_fwd plus_tx.bw  --bigwig_rev minus_tx.bw

    Both flags must be provided together. When supplied, Phase 4.5
    (terminal-exon UTR extension) and Step 5g.5 (downstream-exon
    recovery) veto candidate regions where stranded data shows positive
    antisense dominance (antisense >= 1.0 AND antisense >= 5x sense).
    Sparse stranded data is no-op'd, not vetoed.

    Unstranded libraries cannot be split after the fact; only stranded
    libraries should contribute to these tracks.

Naming options:
    --renumber              Renumber all genes in position order
    --gene_prefix PREFIX    Prefix for gene names (default: GENE)
    --name_from ref.gff     Inherit gene names from this GFF for overlapping
                            genes; new genes get new sequential names

All GFF inputs (--helixer, --stringtie, --transdecoder, --manual_annotation,
--refine_existing) are optional, but at least one must be provided.

Author: Claude (Anthropic) - Gene Annotation Pipeline
"""

import argparse
import configparser
import sys
import os
import re
import logging
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
try:
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d
except ImportError:
    np = None  # check_dependencies() will report this


# ============================================================================
# Spatial index for fast overlap queries
# ============================================================================
class EvidenceIndex:
    """Interval-based spatial index for fast overlap queries against gene models.

    Replaces O(n) linear scans through all evidence genes with O(log n + k)
    lookups where k is the number of overlapping features. Built once at
    pipeline startup, then queried thousands of times during scoring.

    Indexes three things per (seqid, strand):
      - exon intervals from all evidence sources
      - intron intervals (derived from consecutive exon pairs)
      - consecutive exon pairs (for chimeric exon detection)
    """

    def __init__(self):
        # Exon index: (seqid, strand) -> sorted list of (start, end, source_label)
        self._exons = defaultdict(list)
        # Intron index: (seqid, strand) -> sorted list of (start, end, source_label)
        self._introns = defaultdict(list)
        # Consecutive exon pair index for chimeric detection:
        # (seqid, strand) -> set of (e1_start, e1_end, e2_start, e2_end)
        self._exon_pairs = defaultdict(set)
        # CDS start positions: (seqid, strand) -> sorted list of (genomic_pos, source_label)
        # For + strand: start of first CDS segment.  For - strand: end of last CDS segment.
        # These are the genomic positions of the translation initiation codon.
        self._cds_starts = defaultdict(list)
        self._built = False

    def add_genes(self, genes: List['Gene'], source_label: str):
        """Add all exons and introns from a list of gene models."""
        for gene in genes:
            key = (gene.seqid, gene.strand)
            for tx in gene.transcripts:
                sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                for exon in sorted_exons:
                    self._exons[key].append((exon.start, exon.end, source_label))
                # Add CDS as exon evidence too
                for cds in tx.cds:
                    self._exons[key].append((cds.start, cds.end, source_label))
                # Record CDS start (translation initiation site)
                if tx.cds:
                    sorted_cds = sorted(tx.cds, key=lambda c: c.start)
                    if gene.strand == '+':
                        cds_start_pos = sorted_cds[0].start
                    else:
                        cds_start_pos = sorted_cds[-1].end
                    self._cds_starts[key].append((cds_start_pos, source_label))
                # Derive introns from consecutive exons
                for i in range(len(sorted_exons) - 1):
                    intron_start = sorted_exons[i].end + 1
                    intron_end = sorted_exons[i + 1].start - 1
                    if intron_end >= intron_start:
                        self._introns[key].append(
                            (intron_start, intron_end, source_label))
                    # Store consecutive exon pairs
                    self._exon_pairs[key].add((
                        sorted_exons[i].start, sorted_exons[i].end,
                        sorted_exons[i + 1].start, sorted_exons[i + 1].end))

    def build(self):
        """Sort all interval lists for binary search. Call once after all add_genes()."""
        for key in self._exons:
            self._exons[key].sort()
        for key in self._introns:
            self._introns[key].sort()
        for key in self._cds_starts:
            self._cds_starts[key].sort()
        self._built = True
        n_exons = sum(len(v) for v in self._exons.values())
        n_introns = sum(len(v) for v in self._introns.values())
        n_pairs = sum(len(v) for v in self._exon_pairs.values())
        logger.info(f"EvidenceIndex built: {n_exons} exon intervals, "
                   f"{n_introns} intron intervals, {n_pairs} exon pairs")

    def has_overlapping_exon(self, seqid: str, strand: str,
                              query_start: int, query_end: int,
                              source: str = None,
                              tolerance: int = 10) -> bool:
        """Check if any indexed exon overlaps the query interval.

        Args:
            source: If set, only match exons from this source (e.g. 'Helixer').
                    If None, match any source.
        """
        key = (seqid, strand)
        intervals = self._exons.get(key, [])
        if not intervals:
            return False

        # Binary search for first interval that could overlap
        import bisect
        # An interval (s, e) overlaps (query_start, query_end) if
        # s <= query_end + tolerance AND e >= query_start - tolerance
        idx = bisect.bisect_left(intervals, (query_start - tolerance - 50000,))
        # Scan forward from approximate position
        # We need to find intervals where start <= query_end + tolerance
        # Starting from idx, check until start > query_end + tolerance
        while idx < len(intervals):
            s, e, src = intervals[idx]
            if s > query_end + tolerance:
                break
            if e >= query_start - tolerance:
                if source is None or src == source:
                    return True
            idx += 1

        return False

    def has_matching_intron(self, seqid: str, strand: str,
                            intron_start: int, intron_end: int,
                            source: str = None,
                            tolerance: int = 10) -> bool:
        """Check if any indexed intron matches within tolerance."""
        key = (seqid, strand)
        intervals = self._introns.get(key, [])
        if not intervals:
            return False

        import bisect
        idx = bisect.bisect_left(intervals, (intron_start - tolerance - 1000,))
        while idx < len(intervals):
            s, e, src = intervals[idx]
            if s > intron_start + tolerance:
                break
            if (abs(s - intron_start) <= tolerance and
                    abs(e - intron_end) <= tolerance):
                if source is None or src == source:
                    return True
            idx += 1

        return False

    def get_overlapping_exons(self, seqid: str, strand: str,
                               query_start: int, query_end: int,
                               tolerance: int = 10) -> List[Tuple[int, int, str]]:
        """Return all indexed exons overlapping the query interval."""
        key = (seqid, strand)
        intervals = self._exons.get(key, [])
        if not intervals:
            return []

        import bisect
        results = []
        idx = bisect.bisect_left(intervals, (query_start - tolerance - 50000,))
        while idx < len(intervals):
            s, e, src = intervals[idx]
            if s > query_end + tolerance:
                break
            if e >= query_start - tolerance:
                results.append((s, e, src))
            idx += 1
        return results

    def has_exon_pair(self, seqid: str, strand: str,
                       e1_start: int, e1_end: int,
                       e2_start: int, e2_end: int,
                       tolerance: int = 10) -> bool:
        """Check if any evidence source has a consecutive exon pair matching these coords."""
        key = (seqid, strand)
        pairs = self._exon_pairs.get(key, set())
        if not pairs:
            return False
        for ps, pe, ns, ne in pairs:
            if (abs(ps - e1_start) <= tolerance and abs(pe - e1_end) <= tolerance
                    and abs(ns - e2_start) <= tolerance and abs(ne - e2_end) <= tolerance):
                return True
        return False

    def count_supported_exons(self, seqid: str, strand: str,
                               exons: List['Feature'],
                               tolerance: int = 10) -> int:
        """Count how many exons have evidence support (any source)."""
        count = 0
        for exon in exons:
            if self.has_overlapping_exon(seqid, strand,
                                         exon.start, exon.end,
                                         tolerance=tolerance):
                count += 1
        return count

    def has_evidence_for_exon(self, seqid: str, strand: str,
                               exon_start: int, exon_end: int,
                               tolerance: int = 10) -> bool:
        """Check if any evidence exon matches (exactly or containing) this exon."""
        overlaps = self.get_overlapping_exons(seqid, strand,
                                               exon_start, exon_end,
                                               tolerance=tolerance)
        for es, ee, src in overlaps:
            # Exact match within tolerance
            if abs(es - exon_start) <= tolerance and abs(ee - exon_end) <= tolerance:
                return True
            # Evidence exon contains query exon
            if es <= exon_start and ee >= exon_end:
                return True
            # Query exon contains evidence exon (if evidence is >10bp)
            if exon_start <= es and exon_end >= ee and (ee - es + 1) > 10:
                return True
        return False

    def get_evidence_cds_starts(self, seqid: str, strand: str,
                                 region_start: int, region_end: int,
                                 tolerance: int = 10) -> Dict[int, int]:
        """Return evidence CDS start positions within a genomic region.

        Returns a dict mapping genomic_position -> number of independent
        evidence sources that place a translation start there.  Positions
        within `tolerance` of each other are merged (the position with more
        sources wins).

        For + strand, CDS start = leftmost CDS position (ATG genomic start).
        For - strand, CDS start = rightmost CDS position (ATG genomic end).
        """
        import bisect
        key = (seqid, strand)
        cds_list = self._cds_starts.get(key, [])
        if not cds_list:
            return {}

        idx = bisect.bisect_left(cds_list, (region_start - tolerance,))
        raw: Dict[int, set] = {}
        while idx < len(cds_list):
            pos, src = cds_list[idx]
            if pos > region_end + tolerance:
                break
            if region_start - tolerance <= pos <= region_end + tolerance:
                # Merge nearby positions
                merged = False
                for existing_pos in list(raw.keys()):
                    if abs(pos - existing_pos) <= tolerance:
                        raw[existing_pos].add(src)
                        merged = True
                        break
                if not merged:
                    raw[pos] = {src}
            idx += 1

        return {pos: len(srcs) for pos, srcs in raw.items()}

# ============================================================================
# Logging setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Scoring configuration
# ============================================================================
class ScoringConfig:
    """All tuneable scoring parameters. Load from INI with from_file(path)."""

    def __init__(self):
        self.exon_helixer_weight = 0.30
        self.exon_transdecoder_weight = 0.25
        self.exon_stringtie_weight = 0.25
        self.exon_coverage_weight = 0.20
        self.exon_helixer_support = 0.85
        self.exon_helixer_nosupport = 0.15
        self.exon_transdecoder_support = 0.80
        self.exon_transdecoder_nosupport = 0.20
        self.exon_stringtie_support = 0.80
        self.exon_stringtie_nosupport = 0.20
        self.exon_cov_high_min = 5.0
        self.exon_ratio_high_min = 2.0
        self.exon_coverage_high = 0.90
        self.exon_cov_med_min = 2.0
        self.exon_ratio_med_min = 1.5
        self.exon_coverage_medium = 0.70
        self.exon_cov_low_min = 0.5
        self.exon_coverage_low = 0.50
        self.exon_coverage_none = 0.20
        self.intron_canonical_score = 0.90
        self.intron_gc_ag_score = 0.70
        self.intron_noncanonical_score = 0.20
        self.intron_motif_weight = 0.60
        self.intron_pwm_weight = 0.40
        self.intron_evidence_base = 0.30
        self.intron_helixer_bonus = 0.35
        self.intron_stringtie_bonus = 0.35
        self.intron_bam_reads_strong = 10
        self.intron_bam_score_strong = 0.95
        self.intron_bam_reads_good = 5
        self.intron_bam_score_good = 0.85
        self.intron_bam_reads_moderate = 2
        self.intron_bam_score_moderate = 0.70
        self.intron_bam_reads_weak = 1
        self.intron_bam_score_weak = 0.55
        self.intron_bam_score_none = 0.15
        self.intron_splice_weight_bam = 0.25
        self.intron_coverage_weight_bam = 0.20
        self.intron_evidence_weight_bam = 0.25
        self.intron_bam_weight = 0.30
        self.intron_splice_weight_nobam = 0.35
        self.intron_coverage_weight_nobam = 0.35
        self.intron_evidence_weight_nobam = 0.30
        self.cds_helixer_weight = 0.30
        self.cds_transdecoder_weight = 0.35
        self.cds_frame_weight = 0.15
        self.cds_coverage_weight = 0.20
        self.cds_helixer_support = 0.85
        self.cds_helixer_nosupport = 0.15
        self.cds_transdecoder_support = 0.90
        self.cds_transdecoder_nosupport = 0.10
        self.cds_frame_valid = 0.80
        self.cds_frame_unknown = 0.50
        self.cds_cov_normaliser = 10.0
        self.cds_cov_zero = 0.10
        self.gene_bam_junction_min_reads = 2
        self.gene_exon_weight_bam = 0.35
        self.gene_intron_weight_bam = 0.25
        self.gene_consistency_weight_bam = 0.20
        self.gene_bam_junction_weight = 0.20
        self.gene_exon_weight_nobam = 0.40
        self.gene_intron_weight_nobam = 0.35
        self.gene_consistency_weight_nobam = 0.25
        self.gene_exon_weight_single = 0.70
        self.gene_consistency_weight_single = 0.30
        self.coding_threshold = 0.20
        self.ncrna_threshold = 0.20
        self.ncrna_min_coverage = 3.0
        self.ncrna_min_fpkm = 0.5
        self.ncrna_min_coverage_low_fpkm = 5.0

    def write_default_config(self, path):
        """Write current configuration as an annotated INI file."""
        d = self.__dict__
        with open(path, 'w') as f:
            f.write("# ============================================================================\n")
            f.write("# Gene Annotation Refiner - Scoring Configuration\n")
            f.write("# ============================================================================\n")
            f.write("#\n")
            f.write("# This file controls all numerical thresholds and weights used in the\n")
            f.write("# posterior probability scoring of gene models. Edit values to tune the\n")
            f.write("# sensitivity/specificity of the annotation pipeline.\n")
            f.write("#\n")
            f.write("# HOW SCORING WORKS\n")
            f.write("# -----------------\n")
            f.write("# Each feature (exon, intron, CDS, gene) receives a posterior probability\n")
            f.write("# between 0.0 and 1.0, computed as a weighted sum of evidence scores:\n")
            f.write("#\n")
            f.write("#   posterior = w1*score1 + w2*score2 + ... + wN*scoreN\n")
            f.write("#\n")
            f.write("# Within each section, weights forming a linear combination must sum to 1.0.\n")
            f.write("# The pipeline validates this at startup and warns if not.\n")
            f.write("#\n")
            f.write("# EVIDENCE SCORES (0.0 - 1.0)\n")
            f.write("# ---------------------------\n")
            f.write("#   support scores:    awarded when an evidence source confirms a feature\n")
            f.write("#   nosupport scores:  awarded when a source does NOT confirm it\n")
            f.write("#   Setting support = nosupport = 0.5 makes a source neutral.\n")
            f.write("#\n")
            f.write("# BAM EVIDENCE\n")
            f.write("# ------------\n")
            f.write("# When --bam is provided, splice junctions are validated against RNA-seq\n")
            f.write("# read alignments. bam_score_none is the penalty for junctions with ZERO\n")
            f.write("# spliced-read support. Lower = harsher penalty. This is the most impactful\n")
            f.write("# parameter when using BAM evidence with Helixer-only gene models.\n")
            f.write("#\n")
            f.write("# FILTER THRESHOLDS\n")
            f.write("# -----------------\n")
            f.write("#   coding_threshold: minimum posterior for protein-coding genes\n")
            f.write("#   ncrna_threshold:  minimum posterior for ncRNA genes\n")
            f.write("#   Lower = more genes retained (higher sensitivity)\n")
            f.write("#   Higher = fewer genes retained (higher specificity)\n")
            f.write("#\n")
            f.write("# TUNING TIPS\n")
            f.write("# -----------\n")
            f.write("# Trust Helixer more:        raise helixer_support, lower nosupport\n")
            f.write("# Require BAM confirmation:  raise bam_weight, lower bam_score_none\n")
            f.write("# Be lenient without BAM:    raise bam_score_none (e.g. 0.40)\n")
            f.write("# Retain more genes:         lower coding_threshold\n")
            f.write("# Penalise unsupported jxns: lower bam_score_none (e.g. 0.05)\n")
            f.write("#\n")
            f.write("# Generate this file:  python gene_annotation_refiner.py --dump_config\n")
            f.write("# Use this file:       python gene_annotation_refiner.py --config my_config.ini ...\n")
            f.write("\n")
            f.write("[exon_scoring]\n")
            f.write(f"helixer_weight      = {d['exon_helixer_weight']:.2f}\n")
            f.write(f"transdecoder_weight = {d['exon_transdecoder_weight']:.2f}\n")
            f.write(f"stringtie_weight    = {d['exon_stringtie_weight']:.2f}\n")
            f.write(f"coverage_weight     = {d['exon_coverage_weight']:.2f}\n")
            f.write(f"helixer_support      = {d['exon_helixer_support']:.2f}\n")
            f.write(f"helixer_nosupport    = {d['exon_helixer_nosupport']:.2f}\n")
            f.write(f"transdecoder_support = {d['exon_transdecoder_support']:.2f}\n")
            f.write(f"transdecoder_nosupport = {d['exon_transdecoder_nosupport']:.2f}\n")
            f.write(f"stringtie_support    = {d['exon_stringtie_support']:.2f}\n")
            f.write(f"stringtie_nosupport  = {d['exon_stringtie_nosupport']:.2f}\n")
            f.write(f"cov_high_min    = {d['exon_cov_high_min']:.1f}\n")
            f.write(f"ratio_high_min  = {d['exon_ratio_high_min']:.1f}\n")
            f.write(f"coverage_high   = {d['exon_coverage_high']:.2f}\n")
            f.write(f"cov_med_min     = {d['exon_cov_med_min']:.1f}\n")
            f.write(f"ratio_med_min   = {d['exon_ratio_med_min']:.1f}\n")
            f.write(f"coverage_medium = {d['exon_coverage_medium']:.2f}\n")
            f.write(f"cov_low_min     = {d['exon_cov_low_min']:.1f}\n")
            f.write(f"coverage_low    = {d['exon_coverage_low']:.2f}\n")
            f.write(f"coverage_none   = {d['exon_coverage_none']:.2f}\n")
            f.write("\n")
            f.write("[intron_scoring]\n")
            f.write(f"canonical_score    = {d['intron_canonical_score']:.2f}\n")
            f.write(f"gc_ag_score        = {d['intron_gc_ag_score']:.2f}\n")
            f.write(f"noncanonical_score = {d['intron_noncanonical_score']:.2f}\n")
            f.write(f"motif_weight = {d['intron_motif_weight']:.2f}\n")
            f.write(f"pwm_weight   = {d['intron_pwm_weight']:.2f}\n")
            f.write(f"evidence_base   = {d['intron_evidence_base']:.2f}\n")
            f.write(f"helixer_bonus   = {d['intron_helixer_bonus']:.2f}\n")
            f.write(f"stringtie_bonus = {d['intron_stringtie_bonus']:.2f}\n")
            f.write(f"bam_reads_strong   = {d['intron_bam_reads_strong']}\n")
            f.write(f"bam_score_strong   = {d['intron_bam_score_strong']:.2f}\n")
            f.write(f"bam_reads_good     = {d['intron_bam_reads_good']}\n")
            f.write(f"bam_score_good     = {d['intron_bam_score_good']:.2f}\n")
            f.write(f"bam_reads_moderate = {d['intron_bam_reads_moderate']}\n")
            f.write(f"bam_score_moderate = {d['intron_bam_score_moderate']:.2f}\n")
            f.write(f"bam_reads_weak     = {d['intron_bam_reads_weak']}\n")
            f.write(f"bam_score_weak     = {d['intron_bam_score_weak']:.2f}\n")
            f.write(f"bam_score_none     = {d['intron_bam_score_none']:.2f}\n")
            f.write(f"splice_weight_bam    = {d['intron_splice_weight_bam']:.2f}\n")
            f.write(f"coverage_weight_bam  = {d['intron_coverage_weight_bam']:.2f}\n")
            f.write(f"evidence_weight_bam  = {d['intron_evidence_weight_bam']:.2f}\n")
            f.write(f"bam_weight           = {d['intron_bam_weight']:.2f}\n")
            f.write(f"splice_weight_nobam   = {d['intron_splice_weight_nobam']:.2f}\n")
            f.write(f"coverage_weight_nobam = {d['intron_coverage_weight_nobam']:.2f}\n")
            f.write(f"evidence_weight_nobam = {d['intron_evidence_weight_nobam']:.2f}\n")
            f.write("\n")
            f.write("[cds_scoring]\n")
            f.write(f"helixer_weight      = {d['cds_helixer_weight']:.2f}\n")
            f.write(f"transdecoder_weight = {d['cds_transdecoder_weight']:.2f}\n")
            f.write(f"frame_weight        = {d['cds_frame_weight']:.2f}\n")
            f.write(f"coverage_weight     = {d['cds_coverage_weight']:.2f}\n")
            f.write(f"helixer_support        = {d['cds_helixer_support']:.2f}\n")
            f.write(f"helixer_nosupport      = {d['cds_helixer_nosupport']:.2f}\n")
            f.write(f"transdecoder_support   = {d['cds_transdecoder_support']:.2f}\n")
            f.write(f"transdecoder_nosupport = {d['cds_transdecoder_nosupport']:.2f}\n")
            f.write(f"frame_valid            = {d['cds_frame_valid']:.2f}\n")
            f.write(f"frame_unknown          = {d['cds_frame_unknown']:.2f}\n")
            f.write(f"cov_normaliser         = {d['cds_cov_normaliser']:.1f}\n")
            f.write(f"cov_zero               = {d['cds_cov_zero']:.2f}\n")
            f.write("\n")
            f.write("[gene_scoring]\n")
            f.write(f"bam_junction_min_reads = {d['gene_bam_junction_min_reads']}\n")
            f.write(f"exon_weight_bam        = {d['gene_exon_weight_bam']:.2f}\n")
            f.write(f"intron_weight_bam      = {d['gene_intron_weight_bam']:.2f}\n")
            f.write(f"consistency_weight_bam = {d['gene_consistency_weight_bam']:.2f}\n")
            f.write(f"bam_junction_weight    = {d['gene_bam_junction_weight']:.2f}\n")
            f.write(f"exon_weight_nobam        = {d['gene_exon_weight_nobam']:.2f}\n")
            f.write(f"intron_weight_nobam      = {d['gene_intron_weight_nobam']:.2f}\n")
            f.write(f"consistency_weight_nobam = {d['gene_consistency_weight_nobam']:.2f}\n")
            f.write(f"exon_weight_single        = {d['gene_exon_weight_single']:.2f}\n")
            f.write(f"consistency_weight_single = {d['gene_consistency_weight_single']:.2f}\n")
            f.write("\n")
            f.write("[filter_thresholds]\n")
            f.write(f"coding_threshold = {d['coding_threshold']:.2f}\n")
            f.write(f"ncrna_threshold  = {d['ncrna_threshold']:.2f}\n")
            f.write("\n")
            f.write("[ncrna_detection]\n")
            f.write(f"min_coverage          = {d['ncrna_min_coverage']:.1f}\n")
            f.write(f"min_fpkm              = {d['ncrna_min_fpkm']:.1f}\n")
            f.write(f"min_coverage_low_fpkm = {d['ncrna_min_coverage_low_fpkm']:.1f}\n")
        logger.info(f"Default configuration written to: {path}")

    @classmethod
    def from_file(cls, path):
        cfg = cls()
        parser = configparser.ConfigParser()
        parser.read(path)
        section_map = {
            "exon_scoring": {"helixer_weight":"exon_helixer_weight","transdecoder_weight":"exon_transdecoder_weight","stringtie_weight":"exon_stringtie_weight","coverage_weight":"exon_coverage_weight","helixer_support":"exon_helixer_support","helixer_nosupport":"exon_helixer_nosupport","transdecoder_support":"exon_transdecoder_support","transdecoder_nosupport":"exon_transdecoder_nosupport","stringtie_support":"exon_stringtie_support","stringtie_nosupport":"exon_stringtie_nosupport","cov_high_min":"exon_cov_high_min","ratio_high_min":"exon_ratio_high_min","coverage_high":"exon_coverage_high","cov_med_min":"exon_cov_med_min","ratio_med_min":"exon_ratio_med_min","coverage_medium":"exon_coverage_medium","cov_low_min":"exon_cov_low_min","coverage_low":"exon_coverage_low","coverage_none":"exon_coverage_none"},
            "intron_scoring": {"canonical_score":"intron_canonical_score","gc_ag_score":"intron_gc_ag_score","noncanonical_score":"intron_noncanonical_score","motif_weight":"intron_motif_weight","pwm_weight":"intron_pwm_weight","evidence_base":"intron_evidence_base","helixer_bonus":"intron_helixer_bonus","stringtie_bonus":"intron_stringtie_bonus","bam_reads_strong":"intron_bam_reads_strong","bam_score_strong":"intron_bam_score_strong","bam_reads_good":"intron_bam_reads_good","bam_score_good":"intron_bam_score_good","bam_reads_moderate":"intron_bam_reads_moderate","bam_score_moderate":"intron_bam_score_moderate","bam_reads_weak":"intron_bam_reads_weak","bam_score_weak":"intron_bam_score_weak","bam_score_none":"intron_bam_score_none","splice_weight_bam":"intron_splice_weight_bam","coverage_weight_bam":"intron_coverage_weight_bam","evidence_weight_bam":"intron_evidence_weight_bam","bam_weight":"intron_bam_weight","splice_weight_nobam":"intron_splice_weight_nobam","coverage_weight_nobam":"intron_coverage_weight_nobam","evidence_weight_nobam":"intron_evidence_weight_nobam"},
            "cds_scoring": {"helixer_weight":"cds_helixer_weight","transdecoder_weight":"cds_transdecoder_weight","frame_weight":"cds_frame_weight","coverage_weight":"cds_coverage_weight","helixer_support":"cds_helixer_support","helixer_nosupport":"cds_helixer_nosupport","transdecoder_support":"cds_transdecoder_support","transdecoder_nosupport":"cds_transdecoder_nosupport","frame_valid":"cds_frame_valid","frame_unknown":"cds_frame_unknown","cov_normaliser":"cds_cov_normaliser","cov_zero":"cds_cov_zero"},
            "gene_scoring": {"bam_junction_min_reads":"gene_bam_junction_min_reads","exon_weight_bam":"gene_exon_weight_bam","intron_weight_bam":"gene_intron_weight_bam","consistency_weight_bam":"gene_consistency_weight_bam","bam_junction_weight":"gene_bam_junction_weight","exon_weight_nobam":"gene_exon_weight_nobam","intron_weight_nobam":"gene_intron_weight_nobam","consistency_weight_nobam":"gene_consistency_weight_nobam","exon_weight_single":"gene_exon_weight_single","consistency_weight_single":"gene_consistency_weight_single"},
            "filter_thresholds": {"coding_threshold":"coding_threshold","ncrna_threshold":"ncrna_threshold"},
            "ncrna_detection": {"min_coverage":"ncrna_min_coverage","min_fpkm":"ncrna_min_fpkm","min_coverage_low_fpkm":"ncrna_min_coverage_low_fpkm"},
        }
        for section, key_map in section_map.items():
            if not parser.has_section(section):
                continue
            for ini_key, attr_name in key_map.items():
                if parser.has_option(section, ini_key):
                    old_val = getattr(cfg, attr_name)
                    raw = parser.get(section, ini_key)
                    if isinstance(old_val, int):
                        setattr(cfg, attr_name, int(raw))
                    else:
                        setattr(cfg, attr_name, float(raw))
        return cfg

    def log_active_config(self):
        logger.info("Active scoring configuration:")
        logger.info("  Exon weights:   Hx=%.2f  TD=%.2f  ST=%.2f  Cov=%.2f",
                     self.exon_helixer_weight, self.exon_transdecoder_weight,
                     self.exon_stringtie_weight, self.exon_coverage_weight)
        logger.info("  Intron weights (BAM):   Splice=%.2f  Cov=%.2f  Ev=%.2f  BAM=%.2f",
                     self.intron_splice_weight_bam, self.intron_coverage_weight_bam,
                     self.intron_evidence_weight_bam, self.intron_bam_weight)
        logger.info("  Intron weights (noBAM): Splice=%.2f  Cov=%.2f  Ev=%.2f",
                     self.intron_splice_weight_nobam, self.intron_coverage_weight_nobam,
                     self.intron_evidence_weight_nobam)
        logger.info("  BAM scores: 10+->%.2f  5+->%.2f  2+->%.2f  1->%.2f  0->%.2f",
                     self.intron_bam_score_strong, self.intron_bam_score_good,
                     self.intron_bam_score_moderate, self.intron_bam_score_weak,
                     self.intron_bam_score_none)
        logger.info("  Gene weights (BAM):   Ex=%.2f  In=%.2f  Con=%.2f  BAMj=%.2f",
                     self.gene_exon_weight_bam, self.gene_intron_weight_bam,
                     self.gene_consistency_weight_bam, self.gene_bam_junction_weight)
        logger.info("  Gene weights (noBAM): Ex=%.2f  In=%.2f  Con=%.2f",
                     self.gene_exon_weight_nobam, self.gene_intron_weight_nobam,
                     self.gene_consistency_weight_nobam)
        logger.info("  CDS weights:  Hx=%.2f  TD=%.2f  Fr=%.2f  Cov=%.2f",
                     self.cds_helixer_weight, self.cds_transdecoder_weight,
                     self.cds_frame_weight, self.cds_coverage_weight)
        logger.info("  Filter:       coding>=%.2f  ncRNA>=%.2f",
                     self.coding_threshold, self.ncrna_threshold)
        logger.info("  ncRNA:        min_cov=%.1f  min_fpkm=%.1f  low_fpkm_cov=%.1f",
                     self.ncrna_min_coverage, self.ncrna_min_fpkm,
                     self.ncrna_min_coverage_low_fpkm)

    def validate_weights(self):
        groups = {
            "exon": [self.exon_helixer_weight, self.exon_transdecoder_weight,
                     self.exon_stringtie_weight, self.exon_coverage_weight],
            "intron (BAM)": [self.intron_splice_weight_bam, self.intron_coverage_weight_bam,
                             self.intron_evidence_weight_bam, self.intron_bam_weight],
            "intron (noBAM)": [self.intron_splice_weight_nobam, self.intron_coverage_weight_nobam,
                               self.intron_evidence_weight_nobam],
            "CDS": [self.cds_helixer_weight, self.cds_transdecoder_weight,
                    self.cds_frame_weight, self.cds_coverage_weight],
            "gene (BAM)": [self.gene_exon_weight_bam, self.gene_intron_weight_bam,
                           self.gene_consistency_weight_bam, self.gene_bam_junction_weight],
            "gene (noBAM)": [self.gene_exon_weight_nobam, self.gene_intron_weight_nobam,
                             self.gene_consistency_weight_nobam],
            "gene (single)": [self.gene_exon_weight_single, self.gene_consistency_weight_single],
        }
        for name, weights in groups.items():
            total = sum(weights)
            if abs(total - 1.0) > 0.01:
                logger.warning(f"  Config: {name} weights sum to {total:.2f} (expected ~1.0)")


# ============================================================================
# Splice site PWM (Position Weight Matrix)
# ============================================================================
# PWMs are computed empirically from StringTie splice junctions at startup.
# Fallback values below are used only if insufficient junctions are found.
# Donor site: 3bp exon + 6bp intron (positions -3 to +6), 9bp total
# Acceptor site: 6bp intron + 3bp exon (positions -6 to +3), 9bp total
DONOR_PWM = None   # Populated at runtime by SplicePWMBuilder
ACCEPTOR_PWM = None  # Populated at runtime by SplicePWMBuilder

# Window sizes for PWM extraction
DONOR_EXON_BP = 3   # bp of exon to include in donor window
DONOR_INTRON_BP = 6  # bp of intron to include in donor window
ACCEPTOR_INTRON_BP = 6
ACCEPTOR_EXON_BP = 3
DONOR_LEN = DONOR_EXON_BP + DONOR_INTRON_BP   # 9
ACCEPTOR_LEN = ACCEPTOR_INTRON_BP + ACCEPTOR_EXON_BP  # 9


class SplicePWMBuilder:
    """Compute Position Weight Matrices from observed splice junctions."""

    PSEUDOCOUNT = 0.01  # Laplace smoothing to avoid log(0)

    def __init__(self, genome: 'GenomeAccess'):
        self.genome = genome

    def build_from_stringtie(self, stringtie_genes: List['Gene'],
                             min_junctions: int = 20,
                             fallback_organism: str = 'drosophila') -> Tuple[dict, dict]:
        """Extract splice junctions from StringTie transcripts and compute PWMs.

        Returns (donor_pwm, acceptor_pwm) as dicts of {base: [prob_per_position]}.
        Only canonical GT-AG junctions are used for training.
        """
        donor_seqs = []
        acceptor_seqs = []

        for gene in stringtie_genes:
            for tx in gene.transcripts:
                exons = sorted(tx.exons, key=lambda e: e.start)
                strand = gene.strand
                if strand not in ('+', '-'):
                    continue

                for i in range(len(exons) - 1):
                    intron_start = exons[i].end + 1
                    intron_end = exons[i + 1].start - 1

                    if intron_end - intron_start < 20:
                        continue  # Skip very short introns

                    # Extract sequences in genomic orientation
                    donor_seq = self.genome.get_sequence(
                        gene.seqid,
                        exons[i].end - DONOR_EXON_BP + 1,
                        exons[i].end + DONOR_INTRON_BP
                    ).upper()
                    acceptor_seq = self.genome.get_sequence(
                        gene.seqid,
                        exons[i + 1].start - ACCEPTOR_INTRON_BP,
                        exons[i + 1].start + ACCEPTOR_EXON_BP - 1
                    ).upper()

                    # For minus strand, swap and reverse-complement
                    if strand == '-':
                        donor_seq, acceptor_seq = (
                            reverse_complement(acceptor_seq),
                            reverse_complement(donor_seq)
                        )

                    # Validate lengths
                    if len(donor_seq) != DONOR_LEN or len(acceptor_seq) != ACCEPTOR_LEN:
                        continue

                    # Only use canonical GT-AG for PWM training
                    donor_di = donor_seq[DONOR_EXON_BP:DONOR_EXON_BP + 2]
                    acceptor_di = acceptor_seq[ACCEPTOR_INTRON_BP - 2:ACCEPTOR_INTRON_BP]
                    if len(donor_di) < 2 or len(acceptor_di) < 2:
                        continue
                    if donor_di == 'GT' and acceptor_di == 'AG':
                        donor_seqs.append(donor_seq)
                        acceptor_seqs.append(acceptor_seq)

        n_junctions = len(donor_seqs)
        logger.info(f"SplicePWMBuilder: {n_junctions} canonical GT-AG junctions "
                    f"extracted from StringTie")

        if n_junctions < min_junctions:
            logger.warning(f"  Only {n_junctions} junctions found (need {min_junctions}); "
                          f"using {fallback_organism} reference PWM as fallback")
            return self._fallback_pwm(fallback_organism)

        donor_pwm = self._seqs_to_pwm(donor_seqs, DONOR_LEN)
        acceptor_pwm = self._seqs_to_pwm(acceptor_seqs, ACCEPTOR_LEN)

        # Log the matrices
        self._log_pwm("Donor", donor_pwm, DONOR_LEN)
        self._log_pwm("Acceptor", acceptor_pwm, ACCEPTOR_LEN)

        return donor_pwm, acceptor_pwm

    def _seqs_to_pwm(self, seqs: List[str], width: int) -> dict:
        """Convert a list of aligned sequences to a frequency PWM."""
        counts = {b: [0] * width for b in 'ACGT'}
        for seq in seqs:
            for i, base in enumerate(seq):
                if base in counts:
                    counts[base][i] += 1

        n = len(seqs)
        pwm = {}
        for base in 'ACGT':
            pwm[base] = [
                (counts[base][i] + self.PSEUDOCOUNT) /
                (n + 4 * self.PSEUDOCOUNT)
                for i in range(width)
            ]
        return pwm

    # ------------------------------------------------------------------ #
    # Built-in reference PWMs                                            #
    #                                                                    #
    # Derived from published splice-site frequency matrices trimmed /    #
    # padded to the pipeline window: 3 exon + 6 intron bp (donor) and   #
    # 6 intron + 3 exon bp (acceptor).  Positions are 0-based:          #
    #   donor:    [e-3 e-2 e-1 | i+1 i+2 i+3 i+4 i+5 i+6]             #
    #   acceptor: [i-6 i-5 i-4 i-3 i-2 i-1 | e+1 e+2 e+3]             #
    #                                                                    #
    # Sources:                                                           #
    #   Drosophila: Reese et al. 1997 (FlyBase splice-site consensus)    #
    #   Human:      Shapiro & Senapathy 1987 / Burge & Karlin 1997      #
    #   Arabidopsis: Reddy 2007 (Plant Cell)                             #
    #                                                                    #
    # Default fallback order: Drosophila → most appropriate for insects. #
    # ------------------------------------------------------------------ #

    _REFERENCE_PWMS = {
        # ── Drosophila melanogaster ───────────────────────────────────────
        # Donor:    consensus MAG|GTAAGT  (positions: exon-3..intron+6)
        # Acceptor: consensus YYYYYYY|AG|G
        'drosophila': {
            'donor': {
                'A': [0.37, 0.34, 0.60, 0.00, 0.00, 0.57, 0.71, 0.09, 0.24],
                'C': [0.21, 0.37, 0.08, 0.00, 0.00, 0.10, 0.07, 0.10, 0.24],
                'G': [0.27, 0.13, 0.27, 1.00, 0.00, 0.18, 0.10, 0.50, 0.27],
                'T': [0.15, 0.16, 0.05, 0.00, 1.00, 0.15, 0.12, 0.31, 0.25],
            },
            'acceptor': {
                'A': [0.22, 0.19, 0.20, 0.18, 0.12, 0.10, 1.00, 0.00, 0.32],
                'C': [0.25, 0.27, 0.27, 0.27, 0.25, 0.20, 0.00, 0.00, 0.27],
                'G': [0.18, 0.17, 0.18, 0.19, 0.17, 0.13, 0.00, 1.00, 0.25],
                'T': [0.35, 0.37, 0.35, 0.36, 0.46, 0.57, 0.00, 0.00, 0.16],
            },
        },
        # ── Homo sapiens ─────────────────────────────────────────────────
        # Donor:    consensus MAG|GTAAGT
        # Acceptor: YYY[n]YYYYYYY|AG|G
        'human': {
            'donor': {
                'A': [0.36, 0.36, 0.59, 0.00, 0.00, 0.62, 0.72, 0.11, 0.25],
                'C': [0.20, 0.36, 0.08, 0.00, 0.00, 0.09, 0.06, 0.11, 0.22],
                'G': [0.29, 0.12, 0.28, 1.00, 0.00, 0.15, 0.09, 0.49, 0.29],
                'T': [0.15, 0.16, 0.05, 0.00, 1.00, 0.14, 0.13, 0.29, 0.24],
            },
            'acceptor': {
                'A': [0.20, 0.18, 0.20, 0.18, 0.13, 0.09, 1.00, 0.00, 0.31],
                'C': [0.27, 0.28, 0.27, 0.25, 0.24, 0.21, 0.00, 0.00, 0.27],
                'G': [0.17, 0.16, 0.17, 0.18, 0.15, 0.12, 0.00, 1.00, 0.26],
                'T': [0.36, 0.38, 0.36, 0.39, 0.48, 0.58, 0.00, 0.00, 0.16],
            },
        },
        # ── Arabidopsis thaliana ─────────────────────────────────────────
        # Plant introns are AT-AC-rich; donor shows preference for AGGT
        'arabidopsis': {
            'donor': {
                'A': [0.38, 0.37, 0.56, 0.00, 0.00, 0.68, 0.65, 0.12, 0.26],
                'C': [0.18, 0.33, 0.10, 0.00, 0.00, 0.06, 0.05, 0.09, 0.21],
                'G': [0.26, 0.11, 0.30, 1.00, 0.00, 0.12, 0.08, 0.46, 0.28],
                'T': [0.18, 0.19, 0.04, 0.00, 1.00, 0.14, 0.22, 0.33, 0.25],
            },
            'acceptor': {
                'A': [0.24, 0.22, 0.24, 0.22, 0.16, 0.11, 1.00, 0.00, 0.33],
                'C': [0.22, 0.24, 0.23, 0.22, 0.22, 0.18, 0.00, 0.00, 0.25],
                'G': [0.17, 0.16, 0.17, 0.18, 0.14, 0.11, 0.00, 1.00, 0.24],
                'T': [0.37, 0.38, 0.36, 0.38, 0.48, 0.60, 0.00, 0.00, 0.18],
            },
        },
    }

    def _fallback_pwm(self, organism: str = 'drosophila') -> Tuple[dict, dict]:
        """Return a reference PWM for the given organism.

        Falls back to a uniform GT/AG-only PWM if the organism name is not
        recognised.  Default organism is 'drosophila' (most appropriate for
        insects).  Other options: 'human', 'arabidopsis'.
        """
        ref = self._REFERENCE_PWMS.get(organism.lower())
        if ref is None:
            logger.warning(f"  Unknown reference organism '{organism}'; "
                           f"using uniform GT/AG fallback")
            return self._uniform_gtag_pwm()

        logger.info(f"  Using {organism} reference splice-site PWMs as fallback")

        # Validate and normalise each column to sum to 1.0
        def _normalise(pwm_dict, length):
            out = {}
            for b in 'ACGT':
                out[b] = list(pwm_dict[b])
            for i in range(length):
                col_sum = sum(out[b][i] for b in 'ACGT')
                if col_sum > 0:
                    for b in 'ACGT':
                        out[b][i] /= col_sum
                else:
                    for b in 'ACGT':
                        out[b][i] = 0.25
            return out

        donor    = _normalise(ref['donor'],    DONOR_LEN)
        acceptor = _normalise(ref['acceptor'], ACCEPTOR_LEN)
        return donor, acceptor

    def _uniform_gtag_pwm(self) -> Tuple[dict, dict]:
        """Uniform PWM with only the invariant GT/AG dinucleotides enforced."""
        donor = {b: [0.25] * DONOR_LEN for b in 'ACGT'}
        for b in 'ACGT':
            donor[b][DONOR_EXON_BP]     = 0.01
            donor[b][DONOR_EXON_BP + 1] = 0.01
        donor['G'][DONOR_EXON_BP]     = 0.97
        donor['T'][DONOR_EXON_BP + 1] = 0.97

        acceptor = {b: [0.25] * ACCEPTOR_LEN for b in 'ACGT'}
        for b in 'ACGT':
            acceptor[b][ACCEPTOR_INTRON_BP - 2] = 0.01
            acceptor[b][ACCEPTOR_INTRON_BP - 1] = 0.01
        acceptor['A'][ACCEPTOR_INTRON_BP - 2] = 0.97
        acceptor['G'][ACCEPTOR_INTRON_BP - 1] = 0.97

        return donor, acceptor

    @staticmethod
    def _log_pwm(name: str, pwm: dict, width: int):
        """Log a PWM in readable format."""
        logger.info(f"  {name} PWM ({width} positions):")
        header = "      " + "  ".join(f"{i+1:5d}" for i in range(width))
        logger.info(header)
        for base in 'ACGT':
            vals = "  ".join(f"{v:5.3f}" for v in pwm[base])
            logger.info(f"    {base}: {vals}")


class ExonEvidenceCalibrator:
    """Train empirical distributions of exon coverage ratios and intron junction
    read counts from well-supported StringTie transcripts.

    Two quantities are calibrated from data:
      - exon_cov / gene_median_cov  (coverage ratio, normalised per-gene so
                                     genes with different expression levels are
                                     directly comparable)
      - intron junction read count  (from the pre-computed junction file)

    Both are stored as sorted empirical CDFs so that any observed value maps
    to a percentile score (0–1) without hard-coded thresholds.

    A composite exon_posterior is a weighted combination of junction and
    coverage percentile scores, with source count as a weak prior.  The 5th
    percentile of training exon posteriors becomes the data-derived drop
    threshold used in _build_consensus Step B.

    If no BAM/junction evidence is available, the junction weight is
    redistributed to coverage.
    """

    MIN_TX_EXONS   = 3    # minimum exons in a transcript to use it for training
    MIN_TRAIN_EXONS = 100  # fall back to simple heuristics if too little data

    # Posterior weights
    W_JUNCTION = 0.40
    W_COVERAGE = 0.40
    W_SOURCE   = 0.20

    def __init__(self, stringtie_genes: List['Gene'],
                 coverage: 'CoverageAccess',
                 bam_evidence):
        self.coverage = coverage
        self.bam = bam_evidence
        self._cov_ratios: List[float] = []   # sorted empirical CDF
        self._junc_counts: List[int] = []    # sorted empirical CDF
        self.drop_threshold: float = 0.20    # posterior threshold (overwritten by data)
        # Minimum mean-junction-reads-per-intron for a multi-exon template to
        # be accepted as a structural scaffold.  Derived as the 0.1th percentile
        # of the distribution across all multi-exon StringTie transcripts.
        # Templates below this are too poorly supported to be reliable guides.
        self.template_min_junction_mean: float = 0.0
        self._trained = False
        self._train(stringtie_genes)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, stringtie_genes: List['Gene']):
        """Collect coverage ratios and junction counts from StringTie data."""
        cov_ratios: List[float] = []
        junc_counts: List[int] = []

        for gene in stringtie_genes:
            for tx in gene.transcripts:
                exons = sorted(tx.exons, key=lambda e: e.start)
                if len(exons) < self.MIN_TX_EXONS:
                    continue

                covs = [self.coverage.get_mean_coverage(gene.seqid, e.start, e.end)
                        for e in exons]
                non_zero = [c for c in covs if c > 0]
                if not non_zero:
                    continue
                gene_median = sorted(non_zero)[len(non_zero) // 2]
                if gene_median <= 0:
                    continue

                for cov in covs:
                    cov_ratios.append(cov / gene_median)

                for i in range(len(exons) - 1):
                    intron_start = exons[i].end + 1
                    intron_end   = exons[i + 1].start - 1
                    if intron_end > intron_start:
                        count = self.bam.count_spliced_reads(
                            gene.seqid, intron_start, intron_end)
                        # Only include confirmed junctions (count > 0) in the
                        # training CDF.  Zero-count introns are scored as 0.0
                        # directly in _compute_posterior, so including them
                        # here would inflate the baseline and let unsupported
                        # introns masquerade as low-but-acceptable.
                        if count > 0:
                            junc_counts.append(count)

        if len(cov_ratios) < self.MIN_TRAIN_EXONS:
            logger.warning(
                f"ExonEvidenceCalibrator: only {len(cov_ratios)} training exons "
                f"(need {self.MIN_TRAIN_EXONS}); using fallback thresholds")
            return

        self._cov_ratios  = sorted(cov_ratios)
        self._junc_counts = sorted(junc_counts)
        self._trained = True

        # Build the transcript-level junction support distribution.
        # Only multi-exon transcripts are included (single-exon transcripts
        # have no introns and would pull the distribution to zero, making
        # percentile-based thresholds meaningless).
        # The 1st percentile of multi-exon transcript means becomes
        # template_min_junction_mean — this gives a data-derived, non-zero
        # floor that scales with sequencing depth.
        tx_junc_means: List[float] = []
        for gene in stringtie_genes:
            for tx in gene.transcripts:
                exons = sorted(tx.exons, key=lambda e: e.start)
                if len(exons) < 2:
                    continue  # single-exon transcripts have no introns to evaluate
                counts = []
                for i in range(len(exons) - 1):
                    intron_s = exons[i].end + 1
                    intron_e = exons[i + 1].start - 1
                    if intron_e > intron_s:
                        counts.append(self.bam.count_spliced_reads(
                            gene.seqid, intron_s, intron_e))
                if counts:
                    tx_junc_means.append(sum(counts) / len(counts))

        if tx_junc_means:
            tx_junc_means.sort()
            idx_1 = max(0, int(len(tx_junc_means) * 0.01) - 1)
            # Hard floor of 1.0: always require at least 1 read per intron on
            # average.  The percentile may be zero if >1% of multi-exon
            # StringTie transcripts have no Portcullis junction match (common
            # due to small coordinate offsets between tools).
            self.template_min_junction_mean = max(1.0, tx_junc_means[idx_1])

        # Compute drop_threshold as the 5th percentile of training exon posteriors.
        # Every training exon is scored with n_sources=1 (conservative: StringTie
        # alone) so the threshold is not inflated by multi-source agreement.
        training_posteriors: List[float] = []
        for gene in stringtie_genes:
            for tx in gene.transcripts:
                exons = sorted(tx.exons, key=lambda e: e.start)
                if len(exons) < self.MIN_TX_EXONS:
                    continue
                covs = [self.coverage.get_mean_coverage(gene.seqid, e.start, e.end)
                        for e in exons]
                non_zero = [c for c in covs if c > 0]
                if not non_zero:
                    continue
                gene_median = sorted(non_zero)[len(non_zero) // 2]
                if gene_median <= 0:
                    continue
                for i, (exon, cov) in enumerate(zip(exons, covs)):
                    flanking = []
                    if i > 0:
                        flanking.append((exons[i - 1].end + 1, exon.start - 1))
                    if i < len(exons) - 1:
                        flanking.append((exon.end + 1, exons[i + 1].start - 1))
                    p = self._compute_posterior(gene.seqid, cov, gene_median,
                                               flanking, n_sources=1)
                    training_posteriors.append(p)

        if training_posteriors:
            training_posteriors.sort()
            idx_5 = max(0, int(len(training_posteriors) * 0.05) - 1)
            self.drop_threshold = training_posteriors[idx_5]

            n_ex = len(self._cov_ratios)
            n_jn = len(self._junc_counts)
            def _pct(data, f):
                return data[max(0, int(len(data) * f) - 1)]
            logger.info(f"ExonEvidenceCalibrator: trained on {n_ex} exons, "
                       f"{n_jn} introns from StringTie")
            logger.info(f"  Coverage ratio  p5={_pct(self._cov_ratios, 0.05):.3f}  "
                       f"p50={_pct(self._cov_ratios, 0.50):.3f}  "
                       f"p95={_pct(self._cov_ratios, 0.95):.3f}")
            if self._junc_counts:
                logger.info(f"  Junction count  p5={_pct(self._junc_counts, 0.05)}  "
                           f"p50={_pct(self._junc_counts, 0.50)}  "
                           f"p95={_pct(self._junc_counts, 0.95)}")
            logger.info(f"  Drop threshold (5th pct of training posteriors): "
                       f"{self.drop_threshold:.4f}")
            logger.info(f"  Template min junction mean (1st pct of multi-exon transcripts): "
                       f"{self.template_min_junction_mean:.2f}")

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _cdf_score(self, value: float, sorted_data: List[float]) -> float:
        """Percentile rank of value in sorted_data (0–1)."""
        if not sorted_data:
            return 0.5
        import bisect
        idx = bisect.bisect_right(sorted_data, value)
        return idx / len(sorted_data)

    def _compute_posterior(self, seqid: str, exon_cov: float,
                           gene_median_cov: float,
                           flanking_introns: List[Tuple[int, int]],
                           n_sources: int) -> float:
        """Core posterior calculation given pre-fetched values."""
        ratio = exon_cov / gene_median_cov if gene_median_cov > 0 else 1.0
        cov_score = self._cdf_score(ratio, self._cov_ratios)

        src_score = min(1.0, n_sources / 3.0)

        if self.bam.available and flanking_introns:
            junc_scores = []
            for js, je in flanking_introns:
                if je > js:
                    count = self.bam.count_spliced_reads(seqid, js, je)
                    # Zero-count introns score 0.0 unconditionally — they have
                    # no junction support regardless of the training CDF shape.
                    if count == 0:
                        junc_scores.append(0.0)
                    else:
                        junc_scores.append(self._cdf_score(count, self._junc_counts))
            if junc_scores:
                junc_score = sum(junc_scores) / len(junc_scores)
                posterior = (self.W_JUNCTION * junc_score +
                             self.W_COVERAGE * cov_score +
                             self.W_SOURCE   * src_score)
            else:
                w_cov = self.W_JUNCTION + self.W_COVERAGE
                posterior = w_cov * cov_score + self.W_SOURCE * src_score
        else:
            # No junction data: redistribute junction weight to coverage
            w_cov = self.W_JUNCTION + self.W_COVERAGE
            posterior = w_cov * cov_score + self.W_SOURCE * src_score

        return min(1.0, max(0.0, posterior))

    def score_exon(self, seqid: str, exon_start: int, exon_end: int,
                   gene_median_cov: float,
                   flanking_introns: List[Tuple[int, int]],
                   n_sources: int) -> float:
        """Compute exon posterior probability (0–1).

        Args:
            seqid:            scaffold / chromosome name
            exon_start/end:   1-based inclusive coordinates
            gene_median_cov:  median coverage across all exons in the gene
                              (normalises for per-gene expression level)
            flanking_introns: list of (intron_start, intron_end) for the
                              introns immediately flanking this exon
            n_sources:        number of GFF sources that predict this exon
        """
        if not self._trained:
            # Fallback: simple coverage-ratio heuristic
            exon_cov = self.coverage.get_mean_coverage(seqid, exon_start, exon_end)
            ratio = exon_cov / gene_median_cov if gene_median_cov > 0 else 1.0
            return min(1.0, max(0.10, min(ratio, 1.0)))

        exon_cov = self.coverage.get_mean_coverage(seqid, exon_start, exon_end)
        return self._compute_posterior(seqid, exon_cov, gene_median_cov,
                                       flanking_introns, n_sources)


@dataclass
class Feature:
    """A single GFF feature (exon, CDS, UTR, etc.)."""
    seqid: str
    source: str
    ftype: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    score: float
    strand: str
    phase: str
    attributes: dict = field(default_factory=dict)

    @property
    def length(self):
        return self.end - self.start + 1


@dataclass
class Transcript:
    """A transcript model with exons, CDS, UTRs."""
    transcript_id: str
    seqid: str
    strand: str
    start: int
    end: int
    source: str
    exons: List[Feature] = field(default_factory=list)
    cds: List[Feature] = field(default_factory=list)
    five_prime_utrs: List[Feature] = field(default_factory=list)
    three_prime_utrs: List[Feature] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def sorted_exons(self):
        return sorted(self.exons, key=lambda e: e.start)

    def sorted_cds(self):
        return sorted(self.cds, key=lambda c: c.start)

    def introns(self):
        """Return list of (start, end) tuples for introns."""
        exons = self.sorted_exons()
        result = []
        for i in range(len(exons) - 1):
            intron_start = exons[i].end + 1
            intron_end = exons[i + 1].start - 1
            if intron_end >= intron_start:
                result.append((intron_start, intron_end))
        return result


@dataclass
class Gene:
    """A gene model with one or more transcripts."""
    gene_id: str
    seqid: str
    strand: str
    start: int
    end: int
    source: str
    transcripts: List[Transcript] = field(default_factory=list)
    posterior: float = 0.0
    gene_type: str = "protein_coding"  # or "ncRNA"
    attributes: dict = field(default_factory=dict)


# ============================================================================
# Gene tracer — follow target genes through every pipeline stage
# ============================================================================
class GeneTracer:
    """Follow selected genes through every pipeline stage.

    Enabled with --trace_gene <id> (repeatable) and/or
    --trace_region <seqid>:<start>-<end> (repeatable).  When enabled,
    snapshots matching genes at every pipeline step and logs the
    merge-evidence breakdown for any gene pair involving a target.

    A gene matches when any of the following is true:
      - gene.gene_id contains a target ID as a substring (exact or partial)
      - any transcript_id contains a target ID
      - gene.attributes['merged_from'] / ['evidence_sources'] contain a
        target ID — catches contributing source-gene IDs after Step 5
        merges and after Step 9 renumbering
      - gene.seqid == target seqid AND gene footprint overlaps target region

    Substring matching lets a single Helixer ID (e.g. "005173") match
    "Apis_helixer_scaffold_2_005173.1" and any refined gene that
    inherited it via the merge trail.  When the refined/renumbered ID
    is all that's known, use --trace_region to follow the locus.

    All output is prefixed with '[TRACE]' for easy grepping.  The tracer
    is a no-op when disabled (the common case): snapshot() returns
    immediately if no targets were registered.
    """

    def __init__(self, gene_ids=None, regions=None):
        self.gene_ids = [g for g in (gene_ids or []) if g]
        self.regions = list(regions or [])  # List[(seqid, start, end)]
        self.enabled = bool(self.gene_ids or self.regions)

    @staticmethod
    def parse_region(s: str):
        """Parse 'seqid:start-end' into (seqid, int start, int end)."""
        if ':' not in s:
            raise ValueError(
                f"--trace_region must be 'seqid:start-end' (got {s!r})")
        seqid, coords = s.split(':', 1)
        if '-' not in coords:
            raise ValueError(
                f"--trace_region must be 'seqid:start-end' (got {s!r})")
        start_s, end_s = coords.split('-', 1)
        return (seqid, int(start_s.replace(',', '')),
                int(end_s.replace(',', '')))

    def matches(self, gene: 'Gene') -> bool:
        if not self.enabled or gene is None:
            return False
        attrs = gene.attributes or {}
        for gid in self.gene_ids:
            if gid in gene.gene_id:
                return True
            if gid in attrs.get('merged_from', ''):
                return True
            if gid in attrs.get('evidence_sources', ''):
                return True
            for tx in getattr(gene, 'transcripts', []) or []:
                if gid in tx.transcript_id:
                    return True
        for rseqid, rstart, rend in self.regions:
            if (gene.seqid == rseqid
                    and gene.end >= rstart and gene.start <= rend):
                return True
        return False

    def region_overlaps(self, seqid: str, start: int, end: int) -> bool:
        """Does a coordinate range overlap any --trace_region?"""
        if not self.enabled:
            return False
        for rseqid, rstart, rend in self.regions:
            if seqid == rseqid and end >= rstart and start <= rend:
                return True
        return False

    def pair_matches(self, gene_a: 'Gene', gene_b: 'Gene') -> bool:
        """True if either gene matches — used to filter merge-evidence logs."""
        return self.matches(gene_a) or self.matches(gene_b)

    def snapshot(self, stage: str, genes, detailed: bool = True) -> None:
        """Log every gene matching any target at this stage.

        When `detailed` is True (default), also dump per-transcript
        exon and CDS coordinate lists.  Set detailed=False for a
        compact one-line-per-gene summary in steps where you only
        care about gene-level boundaries.
        """
        if not self.enabled:
            return
        hits = [g for g in genes if self.matches(g)]
        if not hits:
            logger.info(f"[TRACE] {stage}: no matching genes "
                        f"(of {len(genes)} total)")
            return
        for g in hits:
            self._dump_gene(stage, g, detailed)

    def _dump_gene(self, stage: str, g: 'Gene', detailed: bool) -> None:
        attrs = g.attributes or {}
        n_tx = len(getattr(g, 'transcripts', []) or [])
        ev = attrs.get('evidence_sources', '?')
        mf = attrs.get('merged_from', '')
        head = (f"[TRACE] {stage} | {g.gene_id} | "
                f"{g.seqid}:{g.start}-{g.end}({g.strand}) | "
                f"tx={n_tx} src={ev}")
        if mf:
            head += f" merged_from={mf}"
        logger.info(head)
        if not detailed:
            return
        for tx in getattr(g, 'transcripts', []) or []:
            exons = sorted((e.start, e.end) for e in tx.exons)
            cds = sorted((c.start, c.end) for c in tx.cds)
            logger.info(
                f"[TRACE]   tx={tx.transcript_id} "
                f"exons({len(exons)})={exons} "
                f"cds({len(cds)})={cds}")

    def event(self, stage: str, message: str) -> None:
        """Log a freeform event (no snapshot).  Used by should_merge etc."""
        if not self.enabled:
            return
        logger.info(f"[TRACE] {stage}: {message}")


# ============================================================================
# GFF/GTF Parsers
# ============================================================================
def parse_gff_attributes(attr_string: str, fmt="gff3") -> dict:
    """Parse GFF3 or GTF attribute string into a dictionary."""
    attrs = {}
    if fmt == "gff3":
        for item in attr_string.strip().split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                attrs[key] = value
    else:  # GTF
        for item in attr_string.strip().rstrip(';').split(';'):
            item = item.strip()
            if ' ' in item:
                key, value = item.split(' ', 1)
                attrs[key] = value.strip('"')
    return attrs


def parse_gff_line(line: str, fmt="gff3") -> Optional[Feature]:
    """Parse a single GFF/GTF line into a Feature object."""
    if line.startswith('#') or not line.strip():
        return None
    parts = line.strip().split('\t')
    if len(parts) < 9:
        return None
    score = 0.0
    try:
        score = float(parts[5])
    except (ValueError, TypeError):
        pass
    return Feature(
        seqid=parts[0],
        source=parts[1],
        ftype=parts[2],
        start=int(parts[3]),
        end=int(parts[4]),
        score=score,
        strand=parts[6],
        phase=parts[7],
        attributes=parse_gff_attributes(parts[8], fmt)
    )


def parse_helixer_gff(filepath: str) -> List[Gene]:
    """Parse Helixer GFF3 into gene models."""
    genes = {}
    transcripts = {}

    with open(filepath) as f:
        for line in f:
            feat = parse_gff_line(line, "gff3")
            if feat is None:
                continue

            if feat.ftype == 'gene':
                gid = feat.attributes.get('ID', '')
                gene = Gene(
                    gene_id=gid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source='Helixer',
                    attributes=feat.attributes
                )
                genes[gid] = gene

            elif feat.ftype == 'mRNA':
                tid = feat.attributes.get('ID', '')
                parent = feat.attributes.get('Parent', '')
                tx = Transcript(
                    transcript_id=tid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source='Helixer',
                    attributes=feat.attributes
                )
                transcripts[tid] = tx
                if parent in genes:
                    genes[parent].transcripts.append(tx)

            elif feat.ftype == 'exon':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].exons.append(feat)

            elif feat.ftype == 'CDS':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].cds.append(feat)

            elif feat.ftype == 'five_prime_UTR':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].five_prime_utrs.append(feat)

            elif feat.ftype == 'three_prime_UTR':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].three_prime_utrs.append(feat)

    return list(genes.values())


def parse_transdecoder_gff(filepath: str) -> List[Gene]:
    """Parse TransDecoder GFF3 into gene models.
    Preserves all isoforms (multiple mRNAs per gene)."""
    genes = {}
    transcripts = {}

    with open(filepath) as f:
        for line in f:
            feat = parse_gff_line(line, "gff3")
            if feat is None:
                continue

            if feat.ftype == 'gene':
                gid = feat.attributes.get('ID', '')
                gene = Gene(
                    gene_id=gid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source='TransDecoder',
                    attributes=feat.attributes
                )
                genes[gid] = gene

            elif feat.ftype == 'mRNA':
                tid = feat.attributes.get('ID', '')
                parent = feat.attributes.get('Parent', '')
                tx = Transcript(
                    transcript_id=tid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source='TransDecoder',
                    attributes=feat.attributes
                )
                transcripts[tid] = tx
                if parent in genes:
                    genes[parent].transcripts.append(tx)

            elif feat.ftype == 'exon':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].exons.append(feat)

            elif feat.ftype == 'CDS':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].cds.append(feat)

            elif 'five_prime_UTR' in feat.ftype or 'utr5p' in feat.ftype.lower():
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    feat.ftype = 'five_prime_UTR'
                    transcripts[parent].five_prime_utrs.append(feat)

            elif 'three_prime_UTR' in feat.ftype or 'utr3p' in feat.ftype.lower():
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    feat.ftype = 'three_prime_UTR'
                    transcripts[parent].three_prime_utrs.append(feat)

    return list(genes.values())


def parse_stringtie_gtf(filepath: str) -> List[Gene]:
    """Parse StringTie GTF into gene/transcript models."""
    genes = {}
    transcripts = {}

    with open(filepath) as f:
        for line in f:
            feat = parse_gff_line(line, "gtf")
            if feat is None:
                continue

            gid = feat.attributes.get('gene_id', '')
            tid = feat.attributes.get('transcript_id', '')

            if feat.ftype == 'transcript':
                if gid not in genes:
                    gene = Gene(
                        gene_id=gid, seqid=feat.seqid, strand=feat.strand,
                        start=feat.start, end=feat.end, source='StringTie',
                        attributes=feat.attributes
                    )
                    genes[gid] = gene
                else:
                    genes[gid].start = min(genes[gid].start, feat.start)
                    genes[gid].end = max(genes[gid].end, feat.end)

                tx = Transcript(
                    transcript_id=tid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source='StringTie',
                    attributes=feat.attributes
                )
                transcripts[tid] = tx
                genes[gid].transcripts.append(tx)

            elif feat.ftype == 'exon':
                if tid in transcripts:
                    transcripts[tid].exons.append(feat)

    return list(genes.values())


def parse_generic_gff3(filepath: str, source_label: str = 'Generic') -> List[Gene]:
    """Parse a generic GFF3 file into gene models.

    Works for human-annotated models, previous refined annotations, or any
    standard GFF3 with gene/mRNA/exon/CDS hierarchy.
    """
    genes = {}
    transcripts = {}

    with open(filepath) as f:
        for line in f:
            feat = parse_gff_line(line, "gff3")
            if feat is None:
                continue

            if feat.ftype == 'gene':
                gid = feat.attributes.get('ID', '')
                gene = Gene(
                    gene_id=gid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source=source_label,
                    attributes=feat.attributes
                )
                genes[gid] = gene

            elif feat.ftype in ('mRNA', 'transcript'):
                tid = feat.attributes.get('ID', '')
                parent = feat.attributes.get('Parent', '')
                tx = Transcript(
                    transcript_id=tid, seqid=feat.seqid, strand=feat.strand,
                    start=feat.start, end=feat.end, source=source_label,
                    attributes=feat.attributes
                )
                transcripts[tid] = tx
                if parent in genes:
                    genes[parent].transcripts.append(tx)

            elif feat.ftype == 'exon':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].exons.append(feat)

            elif feat.ftype == 'CDS':
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    transcripts[parent].cds.append(feat)

            elif 'five_prime_UTR' in feat.ftype or 'utr5p' in feat.ftype.lower():
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    feat.ftype = 'five_prime_UTR'
                    transcripts[parent].five_prime_utrs.append(feat)

            elif 'three_prime_UTR' in feat.ftype or 'utr3p' in feat.ftype.lower():
                parent = feat.attributes.get('Parent', '')
                if parent in transcripts:
                    feat.ftype = 'three_prime_UTR'
                    transcripts[parent].three_prime_utrs.append(feat)

    return list(genes.values())


def apply_manual_annotation_confidence(manual_genes: List[Gene],
                                        high_confidence: float = 0.98,
                                        utr_end_confidence: float = 0.50):
    """Mark manual annotation features with appropriate confidence levels.

    All exons in human-annotated models get high confidence. However, the
    outermost ends of UTRs (the terminal positions of 5' and 3' UTRs) do NOT
    get high confidence because annotators often estimate these approximately.

    Specifically:
    - All exons overlapping CDS: high confidence
    - Internal UTR exon boundaries (shared with CDS or intron): high confidence
    - Terminal UTR ends (the outermost start/end of the gene): low confidence
    """
    for gene in manual_genes:
        for tx in gene.transcripts:
            # Determine CDS span
            sorted_cds = sorted(tx.cds, key=lambda c: c.start) if tx.cds else []
            cds_start = sorted_cds[0].start if sorted_cds else None
            cds_end = sorted_cds[-1].end if sorted_cds else None

            for exon in tx.exons:
                # Default: high confidence for exons within CDS span
                exon.score = high_confidence
                exon.attributes['manual_annotation'] = 'true'

            # Mark CDS features as high confidence
            for cds in tx.cds:
                cds.score = high_confidence
                cds.attributes['manual_annotation'] = 'true'

            # Mark UTR features: internal boundaries high, terminal ends low
            for utr in tx.five_prime_utrs:
                utr.attributes['manual_annotation'] = 'true'
                utr.score = high_confidence
                # The outer end of a 5' UTR is approximate
                if gene.strand == '+':
                    # 5' UTR outer end = its start (leftmost)
                    utr.attributes['outer_end_low_confidence'] = 'start'
                else:
                    # Minus strand: 5' UTR outer end = its end (rightmost)
                    utr.attributes['outer_end_low_confidence'] = 'end'

            for utr in tx.three_prime_utrs:
                utr.attributes['manual_annotation'] = 'true'
                utr.score = high_confidence
                # The outer end of a 3' UTR is approximate
                if gene.strand == '+':
                    utr.attributes['outer_end_low_confidence'] = 'end'
                else:
                    utr.attributes['outer_end_low_confidence'] = 'start'

            # Now adjust the terminal exons that correspond to UTR outer ends
            if tx.exons:
                sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                if cds_start is not None:
                    if gene.strand == '+':
                        # First exon may be 5' UTR — its start is approximate
                        if sorted_exons[0].end < cds_start or sorted_exons[0].start < cds_start:
                            sorted_exons[0].attributes['utr_end_low_confidence'] = 'true'
                        # Last exon may be 3' UTR — its end is approximate
                        if sorted_exons[-1].start > cds_end or sorted_exons[-1].end > cds_end:
                            sorted_exons[-1].attributes['utr_end_low_confidence'] = 'true'
                    else:
                        # Minus strand: last exon (highest coord) is 5' UTR outer end
                        if sorted_exons[-1].start > cds_end or sorted_exons[-1].end > cds_end:
                            sorted_exons[-1].attributes['utr_end_low_confidence'] = 'true'
                        # First exon (lowest coord) is 3' UTR outer end
                        if sorted_exons[0].end < cds_start or sorted_exons[0].start < cds_start:
                            sorted_exons[0].attributes['utr_end_low_confidence'] = 'true'
                else:
                    # No CDS — all exons are non-coding; UTR ends are uncertain
                    # Mark first and last exon outer ends as low confidence
                    sorted_exons[0].attributes['utr_end_low_confidence'] = 'true'
                    if len(sorted_exons) > 1:
                        sorted_exons[-1].attributes['utr_end_low_confidence'] = 'true'

    logger.info(f"Applied manual annotation confidence to {len(manual_genes)} genes "
                f"(exon/CDS={high_confidence}, UTR ends={utr_end_confidence})")


def renumber_genes(genes: List[Gene], prefix: str = 'GENE',
                   name_from_genes: Optional[List[Gene]] = None) -> List[Gene]:
    """Renumber genes in position order with optional name inheritance.

    Args:
        genes: List of genes to renumber (sorted by position).
        prefix: Prefix for gene names (e.g. 'OFASC' -> OFASC_000010).
        name_from_genes: If provided, genes overlapping these models inherit
            their names. Non-overlapping genes get new sequential names.
            The numbering scheme descends from this file.

    Returns:
        genes with updated gene_id and transcript_id fields.
    """
    # Sort output genes by position
    genes.sort(key=lambda g: (g.seqid, g.start))

    if name_from_genes is not None:
        # Build index of reference genes for overlap lookup
        ref_by_pos = defaultdict(list)
        for rg in name_from_genes:
            ref_by_pos[(rg.seqid, rg.strand)].append(rg)

        # Determine next available number from existing names
        # Try to extract numeric suffixes from reference gene IDs
        max_num = 0
        num_pattern = re.compile(r'(\d+)$')
        for rg in name_from_genes:
            m = num_pattern.search(rg.gene_id)
            if m:
                max_num = max(max_num, int(m.group(1)))

        # Next available number after the highest in the reference
        next_num = max_num + 1

        # Assign names: overlapping genes inherit, new genes get sequential
        assigned_ref = set()  # track which reference genes have been used
        for gene in genes:
            key = (gene.seqid, gene.strand)
            best_ref = None
            best_overlap = 0

            for rg in ref_by_pos.get(key, []):
                if rg.gene_id in assigned_ref:
                    continue
                overlap_start = max(gene.start, rg.start)
                overlap_end = min(gene.end, rg.end)
                if overlap_end >= overlap_start:
                    overlap = overlap_end - overlap_start + 1
                    min_len = min(gene.end - gene.start + 1,
                                  rg.end - rg.start + 1)
                    # Require substantial overlap (>30% reciprocal)
                    if overlap > min_len * 0.3 and overlap > best_overlap:
                        best_overlap = overlap
                        best_ref = rg

            if best_ref is not None:
                # Inherit the reference gene name
                old_id = gene.gene_id
                gene.gene_id = best_ref.gene_id
                gene.attributes['ID'] = best_ref.gene_id
                gene.attributes['inherited_from'] = best_ref.gene_id
                assigned_ref.add(best_ref.gene_id)
                logger.debug(f"Inherited name: {old_id} -> {gene.gene_id}")
            else:
                # New gene — assign sequential name
                old_id = gene.gene_id
                gene.gene_id = f"{prefix}_{next_num:06d}"
                gene.attributes['ID'] = gene.gene_id
                gene.attributes['novel_gene'] = 'true'
                next_num += 1
                logger.debug(f"New gene name: {old_id} -> {gene.gene_id}")

            # Renumber transcripts
            for j, tx in enumerate(gene.transcripts, 1):
                tx.transcript_id = f"{gene.gene_id}.{j}"

        logger.info(f"Renamed genes: {sum(1 for g in genes if 'inherited_from' in g.attributes)} "
                    f"inherited, {sum(1 for g in genes if 'novel_gene' in g.attributes)} novel")

    else:
        # Simple sequential renumbering
        for i, gene in enumerate(genes, 1):
            gene.gene_id = f"{prefix}_{i:06d}"
            gene.attributes['ID'] = gene.gene_id
            for j, tx in enumerate(gene.transcripts, 1):
                tx.transcript_id = f"{gene.gene_id}.{j}"

        logger.info(f"Renumbered {len(genes)} genes with prefix '{prefix}'")

    return genes


# ============================================================================
# Genome sequence utilities
# ============================================================================

class ORFFinder:
    """Find the longest open reading frame in a transcript sequence
    and map it back to genomic CDS coordinates."""

    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    START_CODON = 'ATG'

    def __init__(self, genome: 'GenomeAccess'):
        self.genome = genome

    def find_best_orf(self, seqid: str, exons: List[Feature],
                      strand: str,
                      min_orf_len: int = 150,
                      coverage=None,
                      evidence_cds_starts: Dict[int, int] = None) -> Optional[Tuple[int, int, int]]:
        """Find the best ORF in a transcript.

        Selection priority (highest to lowest):
          1. Evidence-supported start: ATG maps to a CDS start position
             from Helixer/TransDecoder/etc.  More sources = higher rank.
          2. Exon span: ORFs that cover most of the exons are preferred.
             Real CDS typically starts in the first 1-3 exons and ends in
             the last 1-3 exons.  A short ORF confined to one exon while
             7 others become UTR is almost certainly wrong.
          3. ORF length: among equally-ranked candidates, the longest ORF
             wins (most likely to be the real protein).

        Parameters
        ----------
        coverage : CoverageAccess, optional
            RNA-seq coverage for ATG filtering.
        evidence_cds_starts : dict, optional
            Mapping of genomic CDS start position -> number of evidence
            sources supporting it.  Obtained from EvidenceIndex.

        Returns (orf_start, orf_end, frame) in 0-based transcript
        coordinates, or None.
        """
        COV_MIN_FRACTION = 0.10
        COV_WINDOW = 30

        tx_seq = self._extract_transcript_sequence(seqid, exons, strand)
        if not tx_seq or len(tx_seq) < 3:
            return None

        # Collect all candidate ORFs (start, end, frame) across all frames
        candidates = []
        seq_len = len(tx_seq)
        for i in range(seq_len - 2):
            if tx_seq[i:i+3] != self.START_CODON:
                continue
            frame = i % 3
            j = i + 3
            while j <= seq_len - 3:
                c = tx_seq[j:j+3]
                if c in self.STOP_CODONS:
                    orf_len = j + 3 - i
                    if orf_len >= min_orf_len:
                        candidates.append((i, j + 3, frame))
                    break
                j += 3
            else:
                orf_len = seq_len - i
                if orf_len >= min_orf_len:
                    candidates.append((i, seq_len, frame))

        if not candidates and min_orf_len > 75:
            return self.find_best_orf(seqid, exons, strand,
                                      min_orf_len=75, coverage=coverage,
                                      evidence_cds_starts=evidence_cds_starts)

        if not candidates:
            return None

        sorted_exons_fwd = sorted(exons, key=lambda e: e.start)
        n_exons = len(sorted_exons_fwd)

        # ── Coverage filter ────────────────────────────────────────────────
        if coverage is not None and getattr(coverage, 'available', True):
            cov_scores = {}
            for orf_start, _, _ in candidates:
                g_pos = self._tx_pos_to_genomic(
                    orf_start, sorted_exons_fwd, strand)
                if g_pos is not None:
                    win_s = max(1, g_pos - COV_WINDOW)
                    win_e = g_pos + COV_WINDOW
                    cov_scores[orf_start] = coverage.get_mean_coverage(
                        seqid, win_s, win_e)
                else:
                    cov_scores[orf_start] = 0.0

            max_cov = max(cov_scores.values()) if cov_scores else 0.0
            if max_cov > 0:
                threshold = max_cov * COV_MIN_FRACTION
                candidates = [c for c in candidates
                               if cov_scores.get(c[0], 0.0) >= threshold]

        if not candidates:
            return None

        # ── Evidence match ─────────────────────────────────────────────────
        # Check if each candidate's ATG maps to a known evidence CDS start.
        EVIDENCE_TOL = 10  # bp tolerance for matching ATG to evidence start
        evidence_match = {}  # orf_start -> n_sources (0 = no match)
        if evidence_cds_starts:
            for orf_start, _, _ in candidates:
                g_pos = self._tx_pos_to_genomic(
                    orf_start, sorted_exons_fwd, strand)
                if g_pos is None:
                    evidence_match[orf_start] = 0
                    continue
                best_n = 0
                for ev_pos, n_srcs in evidence_cds_starts.items():
                    if abs(g_pos - ev_pos) <= EVIDENCE_TOL:
                        best_n = max(best_n, n_srcs)
                evidence_match[orf_start] = best_n

        # ── Exon span ─────────────────────────────────────────────────────
        # Compute what fraction of exons each ORF's CDS would cover.
        # An ORF that spans 7/8 exons is almost certainly better than one
        # confined to 1/8.  We compute this by mapping orf_start and
        # orf_end back to genomic coords and counting covered exons.
        def _exon_span_fraction(orf_start: int, orf_end: int) -> float:
            if n_exons <= 1:
                return 1.0
            g_start = self._tx_pos_to_genomic(
                orf_start, sorted_exons_fwd, strand)
            # orf_end is 1 past the last base; map orf_end-1 for the last CDS base
            g_end = self._tx_pos_to_genomic(
                max(0, orf_end - 1), sorted_exons_fwd, strand)
            if g_start is None or g_end is None:
                return 0.0
            cds_lo = min(g_start, g_end)
            cds_hi = max(g_start, g_end)
            covered = sum(1 for e in sorted_exons_fwd
                          if e.end >= cds_lo and e.start <= cds_hi)
            return covered / n_exons

        # ── Rank candidates ────────────────────────────────────────────────
        # Sort key: (evidence_sources DESC, exon_span DESC, orf_length DESC)
        def sort_key(c):
            orf_start, orf_end, _ = c
            ev = evidence_match.get(orf_start, 0)
            span = _exon_span_fraction(orf_start, orf_end)
            length = orf_end - orf_start
            return (-ev, -span, -length)

        candidates.sort(key=sort_key)
        return candidates[0]

    def _tx_pos_to_genomic(self, tx_pos: int,
                            sorted_exons_fwd: List[Feature],
                            strand: str) -> Optional[int]:
        """Convert a 0-based transcript position to a genomic position.

        sorted_exons_fwd must be sorted by start (ascending) regardless of strand.
        For '-' strand, transcript position 0 corresponds to the end of the last exon.
        """
        if strand == '-':
            walk_exons = list(reversed(sorted_exons_fwd))
        else:
            walk_exons = sorted_exons_fwd

        cum = 0
        for exon in walk_exons:
            exon_len = exon.end - exon.start + 1
            if cum + exon_len > tx_pos:
                offset = tx_pos - cum
                if strand == '+':
                    return exon.start + offset
                else:
                    return exon.end - offset
            cum += exon_len
        return None

    def _extract_transcript_sequence(self, seqid: str,
                                      exons: List[Feature],
                                      strand: str) -> str:
        """Get the spliced transcript sequence."""
        sorted_exons = sorted(exons, key=lambda e: e.start)
        parts = []
        for exon in sorted_exons:
            parts.append(self.genome.get_sequence(seqid, exon.start, exon.end))
        tx_seq = ''.join(parts).upper()
        if strand == '-':
            tx_seq = reverse_complement(tx_seq)
        return tx_seq

    def orf_to_genomic_cds(self, seqid: str, exons: List[Feature],
                            strand: str, orf_start: int,
                            orf_end: int) -> List[Feature]:
        """Convert transcript-coordinate ORF to genomic CDS features with phases."""
        sorted_exons = sorted(exons, key=lambda e: e.start)
        if strand == '-':
            sorted_exons = list(reversed(sorted_exons))

        cds_features = []
        tx_pos = 0
        cds_bases_so_far = 0

        for exon in sorted_exons:
            exon_len = exon.end - exon.start + 1
            exon_tx_end = tx_pos + exon_len

            overlap_start = max(tx_pos, orf_start)
            overlap_end = min(exon_tx_end, orf_end)

            if overlap_start < overlap_end:
                if strand == '+':
                    g_start = exon.start + (overlap_start - tx_pos)
                    g_end = exon.start + (overlap_end - tx_pos) - 1
                else:
                    g_end = exon.end - (overlap_start - tx_pos)
                    g_start = exon.end - (overlap_end - tx_pos) + 1

                phase = (3 - (cds_bases_so_far % 3)) % 3

                cds_features.append(Feature(
                    seqid=seqid, source='Refined', ftype='CDS',
                    start=g_start, end=g_end,
                    score=0.0, strand=strand, phase=phase,
                    attributes={'Parent': ''}
                ))
                cds_bases_so_far += overlap_end - overlap_start

            tx_pos = exon_tx_end

        return cds_features

    def reassign_cds(self, gene: 'Gene', coverage=None,
                     evidence_index: 'EvidenceIndex' = None) -> 'Gene':
        """Re-derive CDS for all transcripts by finding the best ORF."""
        for tx in gene.transcripts:
            if not tx.exons:
                continue

            # Query evidence CDS starts in the gene region
            ev_cds_starts = None
            if evidence_index is not None:
                ev_cds_starts = evidence_index.get_evidence_cds_starts(
                    gene.seqid, gene.strand,
                    gene.start, gene.end)

            orf = self.find_best_orf(gene.seqid, tx.exons, gene.strand,
                                     coverage=coverage,
                                     evidence_cds_starts=ev_cds_starts)
            if orf is None:
                tx.cds = []
                continue

            orf_start, orf_end, frame = orf
            new_cds = self.orf_to_genomic_cds(
                gene.seqid, tx.exons, gene.strand, orf_start, orf_end)

            if new_cds:
                tx.cds = new_cds
                tx.five_prime_utrs = []
                tx.three_prime_utrs = []
                self._derive_utrs(tx, gene.strand)
            else:
                tx.cds = []

        return gene

    def _derive_utrs(self, tx: 'Transcript', strand: str):
        """Derive UTR features from exons and CDS."""
        if not tx.cds:
            return
        sorted_cds = sorted(tx.cds, key=lambda c: c.start)
        cds_start = sorted_cds[0].start
        cds_end = sorted_cds[-1].end

        for exon in sorted(tx.exons, key=lambda e: e.start):
            if strand == '+':
                if exon.end < cds_start:
                    tx.five_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='five_prime_UTR',
                        start=exon.start, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
                elif exon.start < cds_start <= exon.end:
                    if cds_start - 1 >= exon.start:
                        tx.five_prime_utrs.append(Feature(
                            seqid=exon.seqid, source='Refined', ftype='five_prime_UTR',
                            start=exon.start, end=cds_start - 1, score=0, strand=strand,
                            phase='.', attributes=exon.attributes))
                if exon.start > cds_end:
                    tx.three_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='three_prime_UTR',
                        start=exon.start, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
                elif exon.start <= cds_end < exon.end:
                    tx.three_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='three_prime_UTR',
                        start=cds_end + 1, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
            else:
                if exon.start > cds_end:
                    tx.five_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='five_prime_UTR',
                        start=exon.start, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
                elif exon.start <= cds_end < exon.end:
                    tx.five_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='five_prime_UTR',
                        start=cds_end + 1, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
                if exon.end < cds_start:
                    tx.three_prime_utrs.append(Feature(
                        seqid=exon.seqid, source='Refined', ftype='three_prime_UTR',
                        start=exon.start, end=exon.end, score=0, strand=strand,
                        phase='.', attributes=exon.attributes))
                elif exon.start < cds_start <= exon.end:
                    if cds_start - 1 >= exon.start:
                        tx.three_prime_utrs.append(Feature(
                            seqid=exon.seqid, source='Refined', ftype='three_prime_UTR',
                            start=exon.start, end=cds_start - 1, score=0, strand=strand,
                            phase='.', attributes=exon.attributes))


# ============================================================================
# Data structures
# ============================================================================
@dataclass

class GenomeAccess:
    """Access genome sequence with coordinate mapping for subregion FASTAs."""

    def __init__(self, fasta_path: str):
        import pyfaidx
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.seqnames = list(self.fasta.keys())
        # Detect if this is a subregion FASTA and compute offset
        self.offsets = {}
        self.seq_to_fasta = {}
        for name in self.seqnames:
            # Look for pattern like lcl|scaffold_1_1000000-2000000
            match = re.search(r'(.+?)_(\d+)-(\d+)$', name)
            if match:
                original_name = match.group(1)
                # FASTA name encodes 1-based start position;
                # offset = start - 1 so that GFF coord X → FASTA pos X - offset
                offset = int(match.group(2)) - 1
                self.offsets[original_name] = offset
                self.seq_to_fasta[original_name] = name
                logger.info(f"Detected subregion FASTA: {original_name} offset={offset} "
                           f"(region {match.group(2)}-{match.group(3)})")
            else:
                self.offsets[name] = 0
                self.seq_to_fasta[name] = name

    def get_sequence(self, seqid: str, start: int, end: int) -> str:
        """Get sequence for GFF coordinates (1-based inclusive)."""
        fasta_name = self.seq_to_fasta.get(seqid, seqid)
        if fasta_name not in self.fasta:
            return ''
        offset = self.offsets.get(seqid, 0)
        # Convert GFF coords to FASTA coords
        fasta_start = start - offset  # 1-based in FASTA
        fasta_end = end - offset
        if fasta_start < 1:
            fasta_start = 1
        seq_len = len(self.fasta[fasta_name])
        if fasta_end > seq_len:
            fasta_end = seq_len
        # pyfaidx uses 0-based slicing
        return str(self.fasta[fasta_name][fasta_start - 1:fasta_end])

    def get_splice_donor(self, seqid: str, exon_end: int, strand: str) -> str:
        """Get donor splice site sequence (last DONOR_EXON_BP of exon +
        first DONOR_INTRON_BP of intron)."""
        if strand == '+':
            return self.get_sequence(seqid,
                                     exon_end - DONOR_EXON_BP + 1,
                                     exon_end + DONOR_INTRON_BP)
        else:
            seq = self.get_sequence(seqid,
                                    exon_end - DONOR_INTRON_BP,
                                    exon_end + DONOR_EXON_BP - 1)
            return reverse_complement(seq)

    def get_splice_acceptor(self, seqid: str, exon_start: int, strand: str) -> str:
        """Get acceptor splice site sequence (last ACCEPTOR_INTRON_BP of intron +
        first ACCEPTOR_EXON_BP of exon)."""
        if strand == '+':
            return self.get_sequence(seqid,
                                     exon_start - ACCEPTOR_INTRON_BP,
                                     exon_start + ACCEPTOR_EXON_BP - 1)
        else:
            seq = self.get_sequence(seqid,
                                    exon_start - ACCEPTOR_EXON_BP + 1,
                                    exon_start + ACCEPTOR_INTRON_BP)
            return reverse_complement(seq)


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'N': 'N', 'n': 'n'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


# Minimum biologically plausible intron size (bp)
MIN_INTRON_SIZE = 30
# Minimum exon size to retain (bp)
MIN_EXON_SIZE = 3


def enforce_canonical_splice_sites(genome: 'GenomeAccess', seqid: str,
                                    exons: List[Feature], strand: str,
                                    max_adjust: int = 5,
                                    bam_evidence=None) -> List[Feature]:
    """Adjust exon boundaries by up to max_adjust bp to find canonical GT-AG splice sites.

    For each internal exon boundary (i.e., not the first 5' or last 3' end),
    check whether shifting the boundary by ±1-max_adjust bp yields a canonical
    GT-AG (or GC-AG) dinucleotide pair. Prefer the smallest adjustment.

    If bam_evidence is provided and the original boundary already has junction
    read support, it is not adjusted — junction data trump splice-site sequence
    preference.  This prevents a nearby GT dinucleotide (never actually used)
    from overriding a real GC-AG site with thousands of supporting reads.
    """
    if len(exons) < 2:
        return exons

    sorted_exons = sorted(exons, key=lambda e: e.start)
    adjusted = [sorted_exons[0]]  # first exon (may adjust its 3' end below)

    for i in range(len(sorted_exons) - 1):
        exon_a = adjusted[-1]  # already-adjusted upstream exon
        exon_b = sorted_exons[i + 1]

        orig_intron_s = exon_a.end + 1
        orig_intron_e = exon_b.start - 1

        # If the original intron boundary already has junction read support,
        # do not adjust it.  A supported GC-AG (or any other) site is real;
        # a nearby GT-AG without reads is not.
        if (bam_evidence is not None and
                getattr(bam_evidence, 'available', False) and
                orig_intron_e > orig_intron_s):
            # Use donor/acceptor queries (not count_spliced_reads) because
            # portcullis stores intron_start as exon_end (0-based), whereas
            # the pipeline stores it as exon_end+1; the offset causes
            # count_spliced_reads(tolerance=0) to miss real junctions.
            if strand == '+':
                orig_reads = bam_evidence.reads_at_donor(seqid, exon_a.end, tolerance=0)
            else:
                orig_reads = bam_evidence.reads_at_acceptor(seqid, exon_b.start, tolerance=0)
            if orig_reads > 0:
                adjusted.append(Feature(
                    seqid=exon_b.seqid, source=exon_b.source, ftype=exon_b.ftype,
                    start=exon_b.start, end=exon_b.end, score=exon_b.score,
                    strand=exon_b.strand, phase=exon_b.phase,
                    attributes=exon_b.attributes
                ))
                continue

        # The intron is between exon_a.end+1 and exon_b.start-1
        best_a_end = exon_a.end
        best_b_start = exon_b.start
        best_score = -1

        for da in range(-max_adjust, max_adjust + 1):
            for db in range(-max_adjust, max_adjust + 1):
                a_end = exon_a.end + da
                b_start = exon_b.start + db
                intron_s = a_end + 1
                intron_e = b_start - 1

                if intron_e - intron_s + 1 < MIN_INTRON_SIZE:
                    continue

                # Ensure adjusted exons remain valid (start <= end)
                if a_end < exon_a.start or b_start > exon_b.end:
                    continue

                # Check splice dinucleotides
                if strand == '+':
                    donor_di = genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                    acceptor_di = genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
                else:
                    donor_di = reverse_complement(
                        genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                    acceptor_di = reverse_complement(
                        genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()

                if len(donor_di) < 2 or len(acceptor_di) < 2:
                    continue
                if donor_di == 'GT' and acceptor_di == 'AG':
                    # Canonical: prefer smallest total adjustment
                    score = 100 - abs(da) - abs(db)
                    if score > best_score:
                        best_score = score
                        best_a_end = a_end
                        best_b_start = b_start
                elif donor_di == 'GC' and acceptor_di == 'AG':
                    score = 50 - abs(da) - abs(db)
                    if score > best_score:
                        best_score = score
                        best_a_end = a_end
                        best_b_start = b_start

        # Safety: revert if adjustment would invert an exon
        if best_a_end < exon_a.start:
            best_a_end = exon_a.end
        if best_b_start > exon_b.end:
            best_b_start = exon_b.start

        if best_score > 0 and (best_a_end != exon_a.end or best_b_start != exon_b.start):
            logger.debug(f"Splice fix: adjusted exon boundary by "
                        f"{best_a_end - exon_a.end}/{best_b_start - exon_b.start} bp "
                        f"at {exon_a.end}/{exon_b.start}")

        # Apply adjustment
        adjusted[-1] = Feature(
            seqid=exon_a.seqid, source=exon_a.source, ftype=exon_a.ftype,
            start=exon_a.start, end=best_a_end, score=exon_a.score,
            strand=exon_a.strand, phase=exon_a.phase, attributes=exon_a.attributes
        )
        adjusted.append(Feature(
            seqid=exon_b.seqid, source=exon_b.source, ftype=exon_b.ftype,
            start=best_b_start, end=exon_b.end, score=exon_b.score,
            strand=exon_b.strand, phase=exon_b.phase, attributes=exon_b.attributes
        ))

    return adjusted


def filter_impossible_introns(exons: List[Feature],
                               min_intron: int = MIN_INTRON_SIZE) -> List[Feature]:
    """Merge exons separated by introns shorter than min_intron bp.

    If two adjacent exons are separated by fewer than min_intron bp,
    merge them into a single exon spanning both.
    """
    if len(exons) < 2:
        return exons

    sorted_exons = sorted(exons, key=lambda e: e.start)
    result = [sorted_exons[0]]

    for exon in sorted_exons[1:]:
        prev = result[-1]
        gap = exon.start - prev.end - 1
        if gap < min_intron:
            # Merge: extend previous exon to cover both
            result[-1] = Feature(
                seqid=prev.seqid, source='Refined', ftype='exon',
                start=prev.start, end=max(prev.end, exon.end),
                score=prev.score, strand=prev.strand, phase=prev.phase,
                attributes=prev.attributes
            )
        else:
            result.append(exon)

    return result


def deduplicate_isoforms(transcripts: List['Transcript']) -> List['Transcript']:
    """Remove transcripts with identical exon structures.

    Two transcripts are identical if they have the same set of exon
    (start, end) coordinates.
    """
    seen = set()
    unique = []
    for tx in transcripts:
        exon_key = tuple(sorted((e.start, e.end) for e in tx.exons))
        if exon_key not in seen:
            seen.add(exon_key)
            unique.append(tx)
    return unique


def trim_zero_coverage_terminal_exons(tx: 'Transcript', coverage: 'CoverageAccess',
                                       seqid: str, strand: str,
                                       min_coverage: float = 1.0) -> 'Transcript':
    """Remove terminal exons (5' and 3' ends) that have no RNA-seq coverage.

    Works from each end inward, stopping at the first exon with coverage
    above min_coverage. Internal zero-coverage exons are NOT removed (they
    may be real exons with low expression).
    """
    if not tx.exons:
        return tx

    sorted_exons = sorted(tx.exons, key=lambda e: e.start)

    # Determine biological 5'→3' order
    if strand == '-':
        sorted_exons = list(reversed(sorted_exons))

    # Trim from 5' end
    first_good = 0
    for i, exon in enumerate(sorted_exons):
        cov = coverage.get_mean_coverage(seqid, exon.start, exon.end)
        if cov >= min_coverage:
            first_good = i
            break
    else:
        # All exons are zero-coverage; keep at least one
        first_good = 0

    # Trim from 3' end
    last_good = len(sorted_exons) - 1
    for i in range(len(sorted_exons) - 1, -1, -1):
        cov = coverage.get_mean_coverage(seqid, sorted_exons[i].start, sorted_exons[i].end)
        if cov >= min_coverage:
            last_good = i
            break

    if first_good > 0 or last_good < len(sorted_exons) - 1:
        trimmed = sorted_exons[first_good:last_good + 1]
        if not trimmed:
            trimmed = sorted_exons  # safety: keep all if trimming would remove everything
        logger.debug(f"Trimmed {first_good} 5'-end and "
                    f"{len(sorted_exons) - 1 - last_good} 3'-end zero-coverage exons")
        tx.exons = trimmed

    return tx


def remove_zero_coverage_internal_exons(tx: 'Transcript', coverage: 'CoverageAccess',
                                         genome: 'GenomeAccess', seqid: str,
                                         strand: str,
                                         min_coverage: float = 1.0) -> 'Transcript':
    """Remove internal exons with zero coverage if removing them yields canonical splice sites.

    Only removes an exon if:
    - It has coverage < min_coverage
    - It is NOT the first or last exon
    - Its neighbors are well-covered (>= 5× min_coverage)
    - The intron formed by skipping this exon has canonical splice sites
    """
    if len(tx.exons) < 3:
        return tx

    sorted_exons = sorted(tx.exons, key=lambda e: e.start)
    keep = [True] * len(sorted_exons)

    for i in range(1, len(sorted_exons) - 1):
        cov = coverage.get_mean_coverage(seqid, sorted_exons[i].start, sorted_exons[i].end)
        if cov >= min_coverage:
            continue

        # Check neighbors have good coverage
        prev_cov = coverage.get_mean_coverage(seqid, sorted_exons[i - 1].start, sorted_exons[i - 1].end)
        next_cov = coverage.get_mean_coverage(seqid, sorted_exons[i + 1].start, sorted_exons[i + 1].end)
        if prev_cov < 5 * min_coverage or next_cov < 5 * min_coverage:
            continue

        # Check if skipping this exon creates a canonical intron
        skip_intron_s = sorted_exons[i - 1].end + 1
        skip_intron_e = sorted_exons[i + 1].start - 1

        if skip_intron_e - skip_intron_s + 1 < MIN_INTRON_SIZE:
            continue

        if strand == '+':
            donor = genome.get_sequence(seqid, skip_intron_s, skip_intron_s + 1).upper()
            acceptor = genome.get_sequence(seqid, skip_intron_e - 1, skip_intron_e).upper()
        else:
            donor = reverse_complement(
                genome.get_sequence(seqid, skip_intron_e - 1, skip_intron_e)).upper()
            acceptor = reverse_complement(
                genome.get_sequence(seqid, skip_intron_s, skip_intron_s + 1)).upper()

        # Skip check if sequence was truncated at scaffold boundary
        if len(donor) < 2 or len(acceptor) < 2:
            pass  # assume canonical at scaffold boundary
        if (donor == 'GT' and acceptor == 'AG') or (donor == 'GC' and acceptor == 'AG'):
            keep[i] = False
            logger.info(f"Removing zero-coverage internal exon "
                       f"{sorted_exons[i].start}-{sorted_exons[i].end} "
                       f"(cov={cov:.1f}, neighbors={prev_cov:.1f}/{next_cov:.1f})")

    if not all(keep):
        tx.exons = [e for e, k in zip(sorted_exons, keep) if k]

    return tx


def validate_all_splice_sites(genome: 'GenomeAccess', seqid: str,
                               exons: List[Feature], strand: str) -> List[Feature]:
    """Remove exons that create non-canonical splice sites and can't be fixed.

    After removing an exon, checks that the resulting intron across the gap
    still has canonical splice sites. If not, tries removing the next exon too.
    """
    if len(exons) < 2:
        return exons

    sorted_exons = sorted(exons, key=lambda e: e.start)
    keep = [True] * len(sorted_exons)

    for i in range(len(sorted_exons) - 1):
        if not keep[i]:
            continue
        intron_s = sorted_exons[i].end + 1
        intron_e = sorted_exons[i + 1].start - 1

        if intron_e - intron_s + 1 < MIN_INTRON_SIZE:
            continue

        if strand == '+':
            donor = genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
            acceptor = genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
        else:
            donor = reverse_complement(
                genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
            acceptor = reverse_complement(
                genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()

        # Skip check if sequence was truncated at scaffold boundary
        if len(donor) < 2 or len(acceptor) < 2:
            pass  # assume canonical at scaffold boundary
        is_canonical = (donor == 'GT' and acceptor == 'AG')
        is_gc_ag = (donor == 'GC' and acceptor == 'AG')

        if not (is_canonical or is_gc_ag):
            # Check if the problem exon has coverage
            this_cov = 0  # we'd need coverage access here
            # For now, just log the issue
            logger.debug(f"Non-canonical splice site at intron {intron_s}-{intron_e}: "
                        f"{donor}...{acceptor}")

    return [e for e, k in zip(sorted_exons, keep) if k]


def has_stop_codon(genome: GenomeAccess, seqid: str, cds_list: List[Feature],
                   strand: str) -> bool:
    """Check if the last CDS ends with a stop codon."""
    if not cds_list:
        return False
    sorted_cds = sorted(cds_list, key=lambda c: c.start)
    if strand == '+':
        last_cds = sorted_cds[-1]
        seq = genome.get_sequence(seqid, last_cds.end - 2, last_cds.end)
    else:
        last_cds = sorted_cds[0]
        seq = reverse_complement(genome.get_sequence(seqid, last_cds.start, last_cds.start + 2))
    return seq.upper() in ('TAA', 'TAG', 'TGA')


def has_start_codon(genome: GenomeAccess, seqid: str, cds_list: List[Feature],
                    strand: str) -> bool:
    """Check if the first CDS starts with ATG."""
    if not cds_list:
        return False
    sorted_cds = sorted(cds_list, key=lambda c: c.start)
    if strand == '+':
        first_cds = sorted_cds[0]
        seq = genome.get_sequence(seqid, first_cds.start, first_cds.start + 2)
    else:
        first_cds = sorted_cds[-1]
        seq = reverse_complement(genome.get_sequence(seqid, first_cds.end - 2, first_cds.end))
    return seq.upper() == 'ATG'


# ============================================================================
# Splice site scoring
# ============================================================================
def score_splice_site(seq: str, pwm: dict) -> float:
    """Score a sequence against a splice site PWM. Returns log-likelihood ratio."""
    if len(seq) != len(list(pwm.values())[0]):
        return -999.0
    score = 0.0
    for i, base in enumerate(seq.upper()):
        if base in pwm:
            prob = pwm[base][i]
            if prob > 0:
                score += np.log2(prob / 0.25)  # log-odds vs uniform background
            else:
                score -= 10.0  # heavy penalty for impossible position
        else:
            score -= 2.0  # unknown base
    return score


def score_donor(seq: str) -> float:
    """Score donor splice site against the empirical PWM."""
    if DONOR_PWM is None:
        return 0.0
    return score_splice_site(seq, DONOR_PWM)


def score_acceptor(seq: str) -> float:
    """Score acceptor splice site against the empirical PWM."""
    if ACCEPTOR_PWM is None:
        return 0.0
    return score_splice_site(seq, ACCEPTOR_PWM)


def _intron_is_canonical(seqid: str, strand: str,
                          intron_s: int, intron_e: int, genome) -> bool:
    """Return True if the intron [intron_s, intron_e] has a canonical GT-AG or
    GC-AG splice pair.  Mirrors the check inside _remove_noncanonical_exons
    but is exposed at module scope so isoform scoring can use it.

    Returns True when the genome is unavailable or the scaffold is truncated,
    to avoid penalising for missing data.
    """
    if genome is None or intron_e - intron_s + 1 < 4:
        return True
    if strand == '+':
        donor = genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
        acceptor = genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
    else:
        donor = reverse_complement(
            genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
        acceptor = reverse_complement(
            genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()
    if len(donor) < 2 or len(acceptor) < 2:
        return True
    return ((donor == 'GT' and acceptor == 'AG') or
            (donor == 'GC' and acceptor == 'AG'))


def _boundary_yields_canonical_splice(seqid: str, strand: str, which: str,
                                        pos: int, genome) -> bool:
    """Return True if this exon boundary yields a canonical splice dinucleotide
    for the adjacent intron.

    which='end'   → intron is to the right (genomic); the 2bp at pos+1..pos+2
                    is the intron's 5' end (+strand donor / -strand acceptor).
    which='start' → intron is to the left (genomic); the 2bp at pos-2..pos-1
                    is the intron's 3' end (+strand acceptor / -strand donor).

    The check is local (doesn't know the other end of the intron), so it's a
    necessary but not sufficient condition — still enough to rule out
    boundaries that can't produce a canonical splice (donor CA, acceptor TT,
    etc.).  Terminal-exon boundaries aren't splice sites at all; this check
    will treat them with a random-sequence bias but the filter only fires when
    candidates disagree, so the rare spurious activation is harmless.
    """
    if genome is None:
        return True
    if which == 'end':
        seq = genome.get_sequence(seqid, pos + 1, pos + 2).upper()
        if len(seq) < 2:
            return True
        if strand == '+':
            return seq in ('GT', 'GC')
        return reverse_complement(seq) == 'AG'
    # 'start'
    seq = genome.get_sequence(seqid, pos - 2, pos - 1).upper()
    if len(seq) < 2:
        return True
    if strand == '+':
        return seq == 'AG'
    return reverse_complement(seq) in ('GT', 'GC')


def _pick_best_exon_boundary(seqid: str, strand: str, which: str,
                              candidates: set, coverage, genome,
                              bam_evidence=None) -> int:
    """Choose the best exon boundary position from a set of candidates.

    Junction reads are the primary discriminator: if any candidate has read
    support in the junction file, only junction-supported candidates are
    considered (coverage + PWM break ties among them).  When no candidate
    has junction reads, the decision falls back to coverage drop + PWM alone.

    This prevents a marginally better PWM score from overriding a real splice
    site that has direct read evidence (e.g. a 4 bp shift caused by a nearby
    canonical dinucleotide that is never actually used).

    Parameters
    ----------
    which : 'start' or 'end'
        Which exon boundary to evaluate.  'start' → left edge (acceptor site);
        'end' → right edge (donor site).
    candidates : set of int
        The boundary positions proposed by the different annotation sources.
    bam_evidence : junction evidence object, optional
        Provides reads_at_donor / reads_at_acceptor queries.
    """
    if len(candidates) == 1:
        return next(iter(candidates))

    FLANK = 20        # bp of intron to measure for the coverage drop
    BODY  = 50        # bp of exon body to use as baseline coverage

    # On + strand: exon end = donor, exon start = acceptor.
    # On - strand: exon end = acceptor, exon start = donor.
    end_is_donor = (strand == '+')

    # ── Step 1: query junction read counts for each candidate ──────────────
    # Use tolerance=0 (exact match) so that read counts don't bleed between
    # nearby candidates.  When two candidates are only a few bp apart (e.g. a
    # GC-AG donor at position X and a GT-AG donor at X+4), a fuzzy window of
    # ±5 bp would attribute the same reads to both, neutralising the junction
    # signal and letting PWM (which favours GT) break the tie incorrectly.
    junction_reads: dict = {}
    if bam_evidence is not None and getattr(bam_evidence, 'available', False):
        for pos in candidates:
            if which == 'end':
                # The intron to the RIGHT of exon.end always starts at pos+1,
                # regardless of strand.  reads_at_donor(pos) finds junctions
                # with intron_start = pos+1.
                reads = bam_evidence.reads_at_donor(seqid, pos, tolerance=0)
            else:
                # The intron to the LEFT of exon.start always ends at pos-1,
                # regardless of strand.  reads_at_acceptor(pos) finds junctions
                # with intron_end = pos-1.
                reads = bam_evidence.reads_at_acceptor(seqid, pos, tolerance=0)
            junction_reads[pos] = reads

    max_junction_reads = max(junction_reads.values()) if junction_reads else 0

    # If any candidate has junction support, restrict to those candidates.
    # This makes junction evidence the authoritative tie-breaker: a real splice
    # site backed by RNA-seq reads always beats a better-looking PWM at an
    # unsupported position.
    if max_junction_reads > 0:
        active_candidates = {p for p, r in junction_reads.items() if r > 0}
    else:
        active_candidates = set(candidates)

    # ── Step 1b: canonical-dinucleotide filter ─────────────────────────────
    # If the active candidates split between those that yield a canonical
    # splice dinucleotide at this boundary and those that don't, drop the
    # non-canonical ones.  This catches cases where two near-duplicate
    # boundaries differ by a few bp and only one lands on a GT/GC/AG site
    # — e.g. 020810's Helixer last exon end (77934859 → GT donor) vs
    # TransDecoder's boundary (77934852 → CA donor).  Without this filter
    # the non-canonical position can win on coverage alone, silently
    # producing a splice site that _remove_noncanonical_exons then deletes
    # the whole exon to repair.
    if len(active_candidates) > 1 and genome is not None:
        canonical = {p for p in active_candidates
                     if _boundary_yields_canonical_splice(
                         seqid, strand, which, p, genome)}
        if canonical and len(canonical) < len(active_candidates):
            active_candidates = canonical

    # Start from the most conservative (smallest) position as baseline.
    if which == 'start':
        sorted_candidates = sorted(active_candidates)
    else:
        sorted_candidates = sorted(active_candidates, reverse=True)

    # ── Step 2: score by coverage drop + PWM (tiebreaker) ──────────────────
    best_pos = None
    best_score = None

    for pos in sorted_candidates:
        if which == 'end':
            body_start   = max(1, pos - BODY)
            body_end     = pos
            intron_start = pos + 1
            intron_end   = pos + FLANK
            if end_is_donor:
                # + strand exon.end = donor in transcript direction.
                # PWM trained on plus-strand window [pos-DONOR_EXON_BP+1, pos+DONOR_INTRON_BP].
                ss_seq    = genome.get_sequence(
                    seqid, pos - DONOR_EXON_BP + 1, pos + DONOR_INTRON_BP)
                pwm_score = score_donor(ss_seq)
            else:
                # − strand exon.end = acceptor in transcript direction.
                # PWM trained on RC([pos-DONOR_EXON_BP+1, pos+DONOR_INTRON_BP]).
                ss_seq    = reverse_complement(genome.get_sequence(
                    seqid, pos - DONOR_EXON_BP + 1, pos + DONOR_INTRON_BP))
                pwm_score = score_acceptor(ss_seq)
        else:
            intron_start = max(1, pos - FLANK)
            intron_end   = pos - 1
            body_start   = pos
            body_end     = pos + BODY
            if end_is_donor:
                # + strand exon.start = acceptor in transcript direction.
                # PWM trained on plus-strand window [pos-ACCEPTOR_INTRON_BP, pos+ACCEPTOR_EXON_BP-1].
                ss_seq    = genome.get_sequence(
                    seqid, pos - ACCEPTOR_INTRON_BP, pos + ACCEPTOR_EXON_BP - 1)
                pwm_score = score_acceptor(ss_seq)
            else:
                # − strand exon.start = donor in transcript direction.
                # PWM trained on RC([pos-ACCEPTOR_INTRON_BP, pos+ACCEPTOR_EXON_BP-1]).
                ss_seq    = reverse_complement(genome.get_sequence(
                    seqid, pos - ACCEPTOR_INTRON_BP, pos + ACCEPTOR_EXON_BP - 1))
                pwm_score = score_donor(ss_seq)

        body_cov   = coverage.get_mean_coverage(seqid, body_start, body_end)
        intron_cov = coverage.get_mean_coverage(seqid, intron_start, intron_end)

        if body_cov > 0.5:
            drop_score = max(0.0, 1.0 - intron_cov / body_cov)
        else:
            drop_score = 0.5

        pwm_norm  = 1.0 / (1.0 + 2.0 ** (-pwm_score))
        composite = 0.6 * drop_score + 0.4 * pwm_norm

        if best_score is None or composite > best_score:
            best_score = composite
            best_pos = pos

    return best_pos


# ============================================================================
# Coverage analysis
# ============================================================================
class CoverageAccess:
    """Access RNA-seq coverage from one or more bigwig files.

    When multiple paths are supplied, per-base values are summed across
    all files (treat them as replicates contributing additively to the
    coverage signal).
    """

    def __init__(self, bigwig_path):
        import pyBigWig
        if isinstance(bigwig_path, str):
            paths = [bigwig_path]
        else:
            paths = list(bigwig_path)
        self.bws = [pyBigWig.open(p) for p in paths]
        self.paths = paths
        # Union of chroms across all files; record per-file lengths for
        # bounds clipping.
        self.chroms = {}
        for bw in self.bws:
            for c, L in bw.chroms().items():
                self.chroms[c] = max(self.chroms.get(c, 0), L)
        # Backwards compatibility -- single primary handle
        self.bw = self.bws[0] if self.bws else None
        self.available = True
        self._mean_cov_cache: dict = {}
        if len(paths) > 1:
            logger.info(f"  CoverageAccess: summing {len(paths)} bigwig files")

    def get_coverage(self, seqid: str, start: int, end: int) -> 'np.ndarray':
        """Get per-bp coverage for a region (0-based coords for bigwig).
        Sums across all loaded bigwig files."""
        if start is None or end is None or start > end or start < 1:
            return np.zeros(0)
        if seqid not in self.chroms:
            return np.zeros(max(0, end - start))
        bw_start = max(0, start - 1)
        bw_end = min(end, self.chroms[seqid])
        if bw_start >= bw_end:
            return np.zeros(max(0, end - start))
        accum = None
        for bw in self.bws:
            chroms = bw.chroms()
            if seqid not in chroms:
                continue
            local_end = min(bw_end, chroms[seqid])
            if local_end <= bw_start:
                continue
            try:
                vals = bw.values(seqid, bw_start, local_end)
                arr = np.array([v if v is not None and not np.isnan(v) else 0.0
                                for v in vals])
            except Exception:
                arr = np.zeros(local_end - bw_start)
            # Pad to the requested length if this file is shorter
            if len(arr) < (bw_end - bw_start):
                arr = np.concatenate([arr,
                                       np.zeros(bw_end - bw_start - len(arr))])
            accum = arr if accum is None else accum + arr
        if accum is None:
            return np.zeros(max(0, bw_end - bw_start))
        return accum

    def get_mean_coverage(self, seqid: str, start: int, end: int) -> float:
        """Get mean coverage for a region. Results are cached to avoid redundant BigWig queries."""
        key = (seqid, start, end)
        cached = self._mean_cov_cache.get(key)
        if cached is not None:
            return cached
        cov = self.get_coverage(seqid, start, end)
        result = float(np.mean(cov)) if len(cov) > 0 else 0.0
        self._mean_cov_cache[key] = result
        return result

    def get_local_coverage_ratio(self, seqid: str, exon_start: int, exon_end: int,
                                  flank: int = 50) -> float:
        """Get ratio of exon coverage to flanking intron coverage."""
        exon_cov = self.get_mean_coverage(seqid, exon_start, exon_end)
        left_cov = self.get_mean_coverage(seqid, max(1, exon_start - flank), exon_start - 1)
        right_cov = self.get_mean_coverage(seqid, exon_end + 1, exon_end + flank)
        flank_cov = (left_cov + right_cov) / 2.0
        if flank_cov < 0.1:
            return exon_cov / 0.1 if exon_cov > 0 else 0.0
        return exon_cov / flank_cov

    def exon_coverage_consistency(self, seqid: str, exons: List[Feature]) -> float:
        """Score how consistent coverage is across exons of a gene (0 to 1).
        Uses coefficient of variation - lower CV = more consistent = higher score."""
        if len(exons) < 2:
            return 1.0
        coverages = []
        for exon in exons:
            cov = self.get_mean_coverage(seqid, exon.start, exon.end)
            coverages.append(max(cov, 0.1))  # floor to avoid div by zero

        coverages = np.array(coverages)
        # Log-transform to handle exponential coverage distributions
        log_cov = np.log1p(coverages)
        cv = np.std(log_cov) / (np.mean(log_cov) + 1e-10)
        # Convert CV to 0-1 score (CV of 0 = perfect consistency = 1.0)
        return max(0.0, 1.0 - cv)

    def intron_coverage_drop(self, seqid: str, exon_end: int, next_exon_start: int) -> float:
        """Score the coverage drop from exon to intron (higher = more clearly an intron)."""
        # Get coverage at exon boundary (last 20bp of exon)
        exon_boundary = self.get_mean_coverage(seqid, max(1, exon_end - 19), exon_end)
        # Get coverage in first 50bp of intron
        intron_start_cov = self.get_mean_coverage(seqid, exon_end + 1,
                                                   min(exon_end + 50, next_exon_start - 1))
        if exon_boundary < 1.0:
            return 0.0
        ratio = intron_start_cov / exon_boundary
        # Score: 1.0 if perfect drop (ratio=0), 0.0 if no drop (ratio=1)
        return max(0.0, 1.0 - ratio)

    def close(self):
        for bw in self.bws:
            try:
                bw.close()
            except Exception:
                pass


class StrandedCoverage:
    """Optional same-strand coverage from forward/reverse bigwigs.

    Used to veto unstranded-coverage support that may actually be
    antisense reads from a neighboring gene. Bigwigs are expected to be
    keyed by transcript strand (e.g. produced via deepTools
    `bamCoverage --filterRNAstrand forward/reverse`).
    """

    def __init__(self, fwd_path: str = None, rev_path: str = None):
        self.fwd = CoverageAccess(fwd_path) if fwd_path else None
        self.rev = CoverageAccess(rev_path) if rev_path else None
        self.available = self.fwd is not None and self.rev is not None

    def sense_mean(self, seqid: str, start: int, end: int, strand: str) -> float:
        """Mean same-strand coverage. Returns None if stranded data unavailable."""
        if not self.available:
            return None
        bw = self.fwd if strand == '+' else self.rev
        return bw.get_mean_coverage(seqid, start, end)

    def antisense_mean(self, seqid: str, start: int, end: int, strand: str) -> float:
        """Mean opposite-strand coverage. Returns None if stranded data unavailable."""
        if not self.available:
            return None
        bw = self.rev if strand == '+' else self.fwd
        return bw.get_mean_coverage(seqid, start, end)

    def close(self):
        if self.fwd:
            self.fwd.close()
        if self.rev:
            self.rev.close()


class NoCoverageAccess:
    """Stub class when no bigwig file is provided.
    Returns neutral/zero values for all coverage queries."""

    def __init__(self):
        self.chroms = {}
        self.available = False

    def get_coverage(self, seqid, start, end):
        return np.zeros(max(0, end - start))

    def get_mean_coverage(self, seqid, start, end):
        return 0.0

    def get_local_coverage_ratio(self, seqid, exon_start, exon_end, flank=50):
        return 0.0

    def exon_coverage_consistency(self, seqid, exons):
        return 0.5  # neutral

    def intron_coverage_drop(self, seqid, exon_end, next_exon_start):
        return 0.5  # neutral

    def close(self):
        pass


# ============================================================================
# Posterior probability computation
# ============================================================================
class PosteriorCalculator:
    """Compute posterior probabilities for gene features using Bayesian integration
    of multiple evidence sources."""

    # Prior weights for different evidence sources
    HELIXER_PRIOR = 0.30      # ML prediction from sequence
    TRANSDECODER_PRIOR = 0.25 # CDS evidence from RNA-seq + ORF finding
    STRINGTIE_PRIOR = 0.15    # RNA-seq transcript assembly
    COVERAGE_PRIOR = 0.15     # Raw RNA-seq coverage
    BAM_PRIOR = 0.15          # Spliced read evidence from BAM

    def __init__(self, genome: GenomeAccess, coverage: CoverageAccess,
                 bam_evidence=None, config=None, evidence_index=None):
        self.genome = genome
        self.coverage = coverage
        self.bam = bam_evidence or NoBAMEvidence()
        self.cfg = config or ScoringConfig()
        self.evidence_index = evidence_index

    def score_exon(self, seqid: str, exon: Feature, strand: str,
                   helixer_support: bool, td_support: bool,
                   stringtie_support: bool) -> float:
        """Compute posterior probability for a single exon."""
        scores = {}

        # 1. Helixer evidence
        scores['helixer'] = 0.85 if helixer_support else 0.15

        # 2. TransDecoder evidence
        scores['transdecoder'] = 0.80 if td_support else 0.20

        # 3. StringTie evidence
        scores['stringtie'] = 0.80 if stringtie_support else 0.20

        # 4. Coverage evidence
        exon_cov = self.coverage.get_mean_coverage(seqid, exon.start, exon.end)
        cov_ratio = self.coverage.get_local_coverage_ratio(seqid, exon.start, exon.end)

        if exon_cov > 5.0 and cov_ratio > 2.0:
            scores['coverage'] = 0.9
        elif exon_cov > 2.0 and cov_ratio > 1.5:
            scores['coverage'] = 0.7
        elif exon_cov > 0.5:
            scores['coverage'] = 0.5
        else:
            scores['coverage'] = 0.2

        # Weighted combination
        posterior = (
            self.HELIXER_PRIOR * scores['helixer'] +
            self.TRANSDECODER_PRIOR * scores['transdecoder'] +
            self.STRINGTIE_PRIOR * scores['stringtie'] +
            self.COVERAGE_PRIOR * scores['coverage']
        )

        return min(1.0, max(0.0, posterior))

    def score_intron(self, seqid: str, intron_start: int, intron_end: int,
                     strand: str, helixer_support: bool,
                     stringtie_support: bool) -> float:
        """Compute posterior probability for an intron (splice junction)."""
        scores = {}

        # 1. Splice site sequence score
        # On + strand: donor at left side (intron_start-1 = exon_left.end),
        #              acceptor at right side (intron_end+1 = exon_right.start).
        # On - strand: donor at right side (intron_end+1 = exon_right.start),
        #              acceptor at left side (intron_start-1 = exon_left.end).
        if strand == '+':
            donor_seq = self.genome.get_splice_donor(seqid, intron_start - 1, strand)
            acceptor_seq = self.genome.get_splice_acceptor(seqid, intron_end + 1, strand)
        else:
            donor_seq = self.genome.get_splice_donor(seqid, intron_end + 1, strand)
            acceptor_seq = self.genome.get_splice_acceptor(seqid, intron_start - 1, strand)
        donor_score = score_donor(donor_seq) if len(donor_seq) == DONOR_LEN else -5.0
        acceptor_score = score_acceptor(acceptor_seq) if len(acceptor_seq) == ACCEPTOR_LEN else -5.0

        # Check for canonical GT-AG
        if strand == '+':
            intron_seq_start = self.genome.get_sequence(seqid, intron_start, intron_start + 1)
            intron_seq_end = self.genome.get_sequence(seqid, intron_end - 1, intron_end)
        else:
            intron_seq_end_rc = self.genome.get_sequence(seqid, intron_start, intron_start + 1)
            intron_seq_start_rc = self.genome.get_sequence(seqid, intron_end - 1, intron_end)
            intron_seq_start = reverse_complement(intron_seq_start_rc)
            intron_seq_end = reverse_complement(intron_seq_end_rc)

        is_canonical = (intron_seq_start.upper() == 'GT' and intron_seq_end.upper() == 'AG')
        is_gc_ag = (intron_seq_start.upper() == 'GC' and intron_seq_end.upper() == 'AG')

        if is_canonical:
            splice_score = 0.9
        elif is_gc_ag:
            splice_score = 0.7
        else:
            splice_score = 0.2

        # Combine with PWM scores
        pwm_combined = (donor_score + acceptor_score) / 20.0 + 0.5  # normalize roughly to 0-1
        pwm_combined = max(0.0, min(1.0, pwm_combined))
        splice_score = 0.6 * splice_score + 0.4 * pwm_combined

        # 2. Coverage drop at splice sites
        cov_drop = self.coverage.intron_coverage_drop(
            seqid, intron_start - 1, intron_end + 1)
        cov_score = 0.5 + 0.5 * cov_drop

        # 3. Evidence support from annotations
        evidence_score = 0.3
        if helixer_support:
            evidence_score += 0.35
        if stringtie_support:
            evidence_score += 0.35

        # 4. BAM spliced-read evidence
        if self.bam.available:
            n_reads = self.bam.count_spliced_reads(seqid, intron_start, intron_end)
            if n_reads >= self.cfg.intron_bam_reads_strong:
                bam_score = self.cfg.intron_bam_score_strong
            elif n_reads >= self.cfg.intron_bam_reads_good:
                bam_score = self.cfg.intron_bam_score_good
            elif n_reads >= self.cfg.intron_bam_reads_moderate:
                bam_score = self.cfg.intron_bam_score_moderate
            elif n_reads >= self.cfg.intron_bam_reads_weak:
                bam_score = self.cfg.intron_bam_score_weak
            else:
                bam_score = self.cfg.intron_bam_score_none
            # 4-way weighted combination
            posterior = (self.cfg.intron_splice_weight_bam * splice_score +
                        self.cfg.intron_coverage_weight_bam * cov_score +
                        self.cfg.intron_evidence_weight_bam * evidence_score +
                        self.cfg.intron_bam_weight * bam_score)
        else:
            # No BAM: 3-way combination
            posterior = (self.cfg.intron_splice_weight_nobam * splice_score +
                        self.cfg.intron_coverage_weight_nobam * cov_score +
                        self.cfg.intron_evidence_weight_nobam * evidence_score)

        return min(1.0, max(0.0, posterior))

    def score_cds(self, seqid: str, cds: Feature, strand: str,
                  helixer_support: bool, td_support: bool) -> float:
        """Compute posterior probability for a CDS feature."""
        scores = {}

        # Helixer CDS support
        scores['helixer'] = 0.85 if helixer_support else 0.15

        # TransDecoder CDS support (strong evidence for CDS specifically)
        scores['transdecoder'] = 0.90 if td_support else 0.10

        # Check reading frame consistency (phase)
        if cds.phase in ('0', '1', '2'):
            scores['frame'] = 0.8
        else:
            scores['frame'] = 0.5

        # Coverage
        cov = self.coverage.get_mean_coverage(seqid, cds.start, cds.end)
        scores['coverage'] = min(1.0, cov / 10.0) if cov > 0 else 0.1

        posterior = (
            0.30 * scores['helixer'] +
            0.35 * scores['transdecoder'] +
            0.15 * scores['frame'] +
            0.20 * scores['coverage']
        )
        return min(1.0, max(0.0, posterior))

    def score_gene(self, gene: Gene, helixer_genes: List[Gene],
                   td_genes: List[Gene], st_genes: List[Gene]) -> float:
        """Compute overall posterior probability for a gene model.
        Scores each transcript and returns the max (best isoform)."""
        if not gene.transcripts:
            return 0.0

        best_score = 0.0
        for tx in gene.transcripts:
            score = self._score_transcript(gene, tx, helixer_genes, td_genes, st_genes)
            best_score = max(best_score, score)
        return best_score

    def _score_transcript(self, gene: Gene, tx: Transcript,
                          helixer_genes: List[Gene],
                          td_genes: List[Gene], st_genes: List[Gene]) -> float:
        """Score a single transcript using the spatial evidence index."""

        # Score all exons
        exon_scores = []
        eidx = self.evidence_index
        for exon in tx.sorted_exons():
            if eidx:
                h_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='Helixer')
                t_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='TransDecoder')
                s_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='StringTie')
            else:
                h_support = any(self._overlaps_exon(exon, hg) for hg in helixer_genes)
                t_support = any(self._overlaps_exon(exon, tg) for tg in td_genes)
                s_support = any(self._overlaps_exon(exon, sg) for sg in st_genes)
            score = self.score_exon(gene.seqid, exon, gene.strand,
                                    h_support, t_support, s_support)
            exon_scores.append(score)

        # Score introns
        intron_scores = []
        for intron_start, intron_end in tx.introns():
            if eidx:
                h_support = eidx.has_matching_intron(
                    gene.seqid, gene.strand, intron_start, intron_end, source='Helixer')
                s_support = eidx.has_matching_intron(
                    gene.seqid, gene.strand, intron_start, intron_end, source='StringTie')
            else:
                h_support = any(self._has_intron(hg, intron_start, intron_end)
                               for hg in helixer_genes)
                s_support = any(self._has_intron(sg, intron_start, intron_end)
                               for sg in st_genes)
            score = self.score_intron(gene.seqid, intron_start, intron_end,
                                      gene.strand, h_support, s_support)
            intron_scores.append(score)

        # Coverage consistency across exons
        consistency = self.coverage.exon_coverage_consistency(gene.seqid, tx.sorted_exons())

        # BAM junction support for introns
        bam_intron_support = 0.5  # neutral default
        if self.bam.available and intron_scores:
            junction_counts = self.bam.get_junction_read_count_for_gene(gene)
            if junction_counts:
                supported = sum(1 for c in junction_counts.values() if c >= self.cfg.gene_bam_junction_min_reads)
                total = len(junction_counts)
                bam_intron_support = supported / total if total > 0 else 0.5
                logger.debug(f"  BAM junction support for {gene.gene_id}: "
                           f"{supported}/{total} introns supported, "
                           f"score={bam_intron_support:.2f}")
            else:
                logger.debug(f"  BAM: no junction counts returned for {gene.gene_id}")

        # Combine scores
        mean_exon = np.mean(exon_scores) if exon_scores else 0.5
        mean_intron = np.mean(intron_scores) if intron_scores else 0.5

        # Overall gene score
        if intron_scores:
            if self.bam.available:
                gene_score = (self.cfg.gene_exon_weight_bam * mean_exon +
                             self.cfg.gene_intron_weight_bam * mean_intron +
                             self.cfg.gene_consistency_weight_bam * consistency +
                             self.cfg.gene_bam_junction_weight * bam_intron_support)
            else:
                gene_score = (self.cfg.gene_exon_weight_nobam * mean_exon +
                             self.cfg.gene_intron_weight_nobam * mean_intron +
                             self.cfg.gene_consistency_weight_nobam * consistency)
        else:
            # Single-exon gene
            gene_score = (self.cfg.gene_exon_weight_single * mean_exon +
                         self.cfg.gene_consistency_weight_single * consistency)

        return min(1.0, max(0.0, gene_score))

    def _overlaps_exon(self, query_exon: Feature, gene: Gene, tolerance: int = 10) -> bool:
        """Check if an exon overlaps any exon in a gene model."""
        for tx in gene.transcripts:
            for exon in tx.exons:
                if (query_exon.start - tolerance <= exon.end and
                    query_exon.end + tolerance >= exon.start):
                    return True
        return False

    def _has_intron(self, gene: Gene, intron_start: int, intron_end: int,
                    tolerance: int = 10) -> bool:
        """Check if a gene has a matching intron."""
        for tx in gene.transcripts:
            for istart, iend in tx.introns():
                if (abs(istart - intron_start) <= tolerance and
                    abs(iend - intron_end) <= tolerance):
                    return True
        return False


# ============================================================================
# Gene merging logic
# ============================================================================
class GeneMerger:
    """Handle merging of gene models that should be combined."""

    def __init__(self, genome: GenomeAccess, coverage: CoverageAccess,
                 bam_evidence=None, st_genes=None, tracer: 'GeneTracer' = None):
        self.genome = genome
        self.coverage = coverage
        self.bam = bam_evidence or NoBAMEvidence()
        self.st_genes = st_genes or []
        self.tracer = tracer or GeneTracer()

    def _trace_ev(self, msg: str, upstream: Gene, downstream: Gene) -> None:
        """Emit merge-evidence line as debug; also to trace log if pair matches."""
        logger.debug(msg)
        if self.tracer.enabled and self.tracer.pair_matches(upstream, downstream):
            logger.info(f"[TRACE] should_merge: {msg}")

    def should_merge(self, gene_a: Gene, gene_b: Gene) -> bool:
        """Determine if two adjacent genes should be merged.

        Considers:
        - Coverage continuity between genes
        - Splice site evidence at gene boundaries
        - Reading frame compatibility
        - Presence of intervening single-exon genes (possible TE artifacts)
        """
        if gene_a.strand != gene_b.strand:
            return False
        if gene_a.seqid != gene_b.seqid:
            return False

        # gene_a is always left of gene_b in genomic coordinates (caller
        # iterates sorted genes).  Check non-overlap in genomic order.
        if gene_a.end >= gene_b.start:
            if self.tracer.enabled and self.tracer.pair_matches(gene_a, gene_b):
                self.tracer.event(
                    "should_merge",
                    f"{gene_a.gene_id} x {gene_b.gene_id}: REJECT "
                    f"(overlapping, not adjacent)")
            return False  # Overlapping, not adjacent

        # Upstream/downstream in transcript direction
        if gene_a.strand == '+':
            upstream, downstream = gene_a, gene_b
        else:
            upstream, downstream = gene_b, gene_a

        # Gap is always between the left gene's end and right gene's start
        # (genomic order, regardless of strand).
        gap_start = gene_a.end + 1
        gap_end = gene_b.start - 1
        gap_size = gap_end - gap_start + 1

        if gap_size <= 0 or gap_size > 50000:
            if self.tracer.enabled and self.tracer.pair_matches(gene_a, gene_b):
                self.tracer.event(
                    "should_merge",
                    f"{gene_a.gene_id} x {gene_b.gene_id}: REJECT "
                    f"(gap_size={gap_size} outside [1, 50000])")
            return False

        # Veto the merge if the gap would become a non-canonical intron.
        # The merge treats the gap between the two genes' terminal exons as
        # a new intron.  If its donor/acceptor are not GT/GC...AG, this is a
        # fabricated intron that _remove_noncanonical_exons will later try
        # to "repair" by cascade-deleting internal exons, wiping the gene.
        # Why: see gene g3005702 + Helixer_001561 merge — gap AA...GA
        # caused 10 of 14 exons to be removed and the transcript dropped.
        # How to apply: any gap larger than MIN_INTRON_SIZE must have
        # canonical splice dinucleotides on the transcribed strand.
        if gap_size >= MIN_INTRON_SIZE:
            if upstream.strand == '+':
                donor_di = self.genome.get_sequence(
                    upstream.seqid, gap_start, gap_start + 1).upper()
                acceptor_di = self.genome.get_sequence(
                    upstream.seqid, gap_end - 1, gap_end).upper()
            else:
                donor_di = reverse_complement(self.genome.get_sequence(
                    upstream.seqid, gap_end - 1, gap_end)).upper()
                acceptor_di = reverse_complement(self.genome.get_sequence(
                    upstream.seqid, gap_start, gap_start + 1)).upper()
            if (len(donor_di) >= 2 and len(acceptor_di) >= 2 and
                    (donor_di not in ('GT', 'GC') or acceptor_di != 'AG')):
                if self.tracer.enabled and self.tracer.pair_matches(
                        upstream, downstream):
                    self.tracer.event(
                        "should_merge",
                        f"{upstream.gene_id} x {downstream.gene_id}: REJECT "
                        f"(merge-gap intron non-canonical "
                        f"donor={donor_di} acceptor={acceptor_di})")
                return False

        # Prospective UTR-exon check: if the merge would leave >=3 exons in
        # the "middle zone" between the two donors' CDS regions, each becomes
        # a UTR exon of the combined gene.  Real genes almost never carry
        # that many UTR exons on one side of their CDS, so this pattern is a
        # strong signal of two neighboring genes being wrongly joined.  This
        # is the cheap prospective counterpart of Step 5h.5's excessive-UTR
        # split and avoids paying for the merge + later split round-trip.
        u_tx = upstream.transcripts[0] if upstream.transcripts else None
        d_tx = downstream.transcripts[0] if downstream.transcripts else None
        if u_tx and u_tx.cds and d_tx and d_tx.cds:
            u_cds_min = min(c.start for c in u_tx.cds)
            u_cds_max = max(c.end for c in u_tx.cds)
            d_cds_min = min(c.start for c in d_tx.cds)
            d_cds_max = max(c.end for c in d_tx.cds)
            if upstream.strand == '+':
                u_middle = sum(1 for e in u_tx.exons if e.start > u_cds_max)
                d_middle = sum(1 for e in d_tx.exons if e.end < d_cds_min)
            else:
                u_middle = sum(1 for e in u_tx.exons if e.end < u_cds_min)
                d_middle = sum(1 for e in d_tx.exons if e.start > d_cds_max)
            middle_utr = u_middle + d_middle
            if middle_utr >= 3:
                if self.tracer.enabled and self.tracer.pair_matches(
                        upstream, downstream):
                    self.tracer.event(
                        "should_merge",
                        f"{upstream.gene_id} x {downstream.gene_id}: REJECT "
                        f"(would produce {middle_utr} middle-zone UTR exons; "
                        f"up_post_cds={u_middle} down_pre_cds={d_middle})")
                return False

        evidence_count = 0
        evidence_required = 2
        # Require more evidence for larger gaps
        if gap_size > 20000:
            evidence_required = 3

        if self.tracer.enabled and self.tracer.pair_matches(upstream, downstream):
            self.tracer.event(
                "should_merge",
                f"evaluating {upstream.gene_id} -> {downstream.gene_id} "
                f"gap={gap_size} required={evidence_required}")

        # Check 1: Does the upstream gene lack a stop codon?
        if upstream.transcripts:
            tx = upstream.transcripts[0]
            if tx.cds and not has_stop_codon(self.genome, upstream.seqid, tx.cds, upstream.strand):
                evidence_count += 1
                self._trace_ev(
                    f"Merge evidence: {upstream.gene_id} lacks stop codon",
                    upstream, downstream)

        # Check 2: Does the upstream gene end with a splice donor site?
        # For + strand the 3' terminal exon is exons[-1] and the donor
        # boundary is at exon.end.  For - strand the 3' terminal exon is
        # exons[0] and the donor boundary is at exon.start (intron departs
        # to the left in genomic coords).
        if upstream.transcripts:
            tx = upstream.transcripts[0]
            exons = tx.sorted_exons()
            if exons:
                if upstream.strand == '+':
                    donor_boundary = exons[-1].end
                else:
                    donor_boundary = exons[0].start
                donor_seq = self.genome.get_splice_donor(
                    upstream.seqid, donor_boundary, upstream.strand)
                if len(donor_seq) == 9:
                    donor_s = score_donor(donor_seq)
                    if donor_s > 2.0:  # Good splice site
                        evidence_count += 1
                        self._trace_ev(
                            f"Merge evidence: {upstream.gene_id} ends with "
                            f"splice donor (score={donor_s:.2f})",
                            upstream, downstream)

        # Check 3: Reading frame compatibility
        if upstream.transcripts and downstream.transcripts:
            up_tx = upstream.transcripts[0]
            down_tx = downstream.transcripts[0]
            if up_tx.cds and down_tx.cds:
                frame_compatible = self._check_frame_compatibility(
                    up_tx, down_tx, upstream.strand)
                if frame_compatible:
                    evidence_count += 1
                    self._trace_ev(
                        f"Merge evidence: Frame compatible between "
                        f"{upstream.gene_id} and {downstream.gene_id}",
                        upstream, downstream)

        # Check 4: Coverage bridge between genes.
        # Valid coverage bridges have gap coverage roughly consistent with
        # the flanking genes (30% - 300% of gene coverage).  If gap coverage
        # is dramatically higher than gene coverage, the gap contains a
        # separate, more highly-expressed gene (not a bridge) and this should
        # VETO the merge rather than support it.
        gap_has_other_gene = False
        if gap_size > 0 and gap_size < 50000:
            gap_cov = self.coverage.get_mean_coverage(
                upstream.seqid, gap_start, gap_end)
            upstream_cov = self.coverage.get_mean_coverage(
                upstream.seqid, max(1, upstream.end - 100), upstream.end)
            downstream_cov = self.coverage.get_mean_coverage(
                upstream.seqid, downstream.start, min(downstream.start + 100, downstream.end))
            avg_gene_cov = (upstream_cov + downstream_cov) / 2.0

            if avg_gene_cov > 1.0:
                ratio = gap_cov / avg_gene_cov
                if ratio > 3.0:
                    gap_has_other_gene = True
                    self._trace_ev(
                        f"Merge vetoed: gap coverage ({gap_cov:.1f}) is "
                        f"{ratio:.1f}x higher than gene coverage "
                        f"({avg_gene_cov:.1f}) — likely separate gene in gap "
                        f"({upstream.gene_id} <-> {downstream.gene_id})",
                        upstream, downstream)
                elif ratio > 0.3:
                    evidence_count += 1
                    self._trace_ev(
                        f"Merge evidence: Coverage bridge "
                        f"(gap_cov={gap_cov:.1f}, gene_cov={avg_gene_cov:.1f})",
                        upstream, downstream)

        if gap_has_other_gene:
            return False

        # Check 5: BAM spliced reads connecting the two genes
        if self.bam.available:
            # Look for any splice junction connecting an exon in upstream to
            # an exon in downstream
            junctions = self.bam.find_novel_junctions(
                upstream.seqid, upstream.start, downstream.end, min_reads=2)
            for j_start, j_end, j_count in junctions:
                # Does this junction span from upstream gene region to downstream?
                if (upstream.start <= j_start <= upstream.end and
                        downstream.start <= j_end <= downstream.end):
                    evidence_count += 1
                    self._trace_ev(
                        f"Merge evidence: BAM spliced reads connect genes "
                        f"({j_count} reads, junction {j_start}-{j_end})",
                        upstream, downstream)
                    break

        # Need sufficient evidence to merge
        if evidence_count < evidence_required:
            if self.tracer.enabled and self.tracer.pair_matches(upstream, downstream):
                self.tracer.event(
                    "should_merge",
                    f"{upstream.gene_id} x {downstream.gene_id}: REJECT "
                    f"(evidence {evidence_count} < required {evidence_required})")
            return False

        # Veto merge if StringTie models the two regions as separate genes
        # AND no StringTie transcript spans both genes
        st_genes_in_a = [g for g in self.st_genes
                         if g.seqid == upstream.seqid
                         and g.start <= upstream.end and g.end >= upstream.start]
        st_genes_in_b = [g for g in self.st_genes
                         if g.seqid == downstream.seqid
                         and g.start <= downstream.end and g.end >= downstream.start]

        if st_genes_in_a and st_genes_in_b:
            # Both have StringTie models — check if any ST gene spans both
            st_spanning = [g for g in self.st_genes
                          if g.seqid == upstream.seqid
                          and g.start <= upstream.start + 500
                          and g.end >= downstream.end - 500]
            if not st_spanning:
                # StringTie treats them as separate -> veto merge
                a_ids = set(g.gene_id for g in st_genes_in_a)
                b_ids = set(g.gene_id for g in st_genes_in_b)
                if not a_ids & b_ids:  # different ST gene IDs
                    self._trace_ev(
                        f"Merge vetoed: StringTie models {upstream.gene_id} "
                        f"and {downstream.gene_id} as separate genes",
                        upstream, downstream)
                    return False

        # Check coverage continuity in the gap — require >50% of flanking coverage
        if gap_size > 100:
            gap_cov = self.coverage.get_mean_coverage(upstream.seqid, gap_start, gap_end)
            up_cov = self.coverage.get_mean_coverage(
                upstream.seqid, max(upstream.start, upstream.end - 500), upstream.end)
            down_cov = self.coverage.get_mean_coverage(
                downstream.seqid, downstream.start, min(downstream.end, downstream.start + 500))
            flank_cov = (up_cov + down_cov) / 2 if (up_cov + down_cov) > 0 else 1
            if gap_cov < flank_cov * 0.1:
                self._trace_ev(
                    f"Merge vetoed: coverage gap ({gap_cov:.1f} vs "
                    f"flanking {flank_cov:.1f})",
                    upstream, downstream)
                return False

        if self.tracer.enabled and self.tracer.pair_matches(upstream, downstream):
            self.tracer.event(
                "should_merge",
                f"{upstream.gene_id} x {downstream.gene_id}: ACCEPT "
                f"(evidence {evidence_count}/{evidence_required}, gap={gap_size})")
        return True

    def _check_frame_compatibility(self, tx_a: Transcript, tx_b: Transcript,
                                    strand: str) -> bool:
        """Check if merging two transcripts maintains reading frame."""
        cds_a = sorted(tx_a.cds, key=lambda c: c.start)
        cds_b = sorted(tx_b.cds, key=lambda c: c.start)

        if not cds_a or not cds_b:
            return False

        if strand == '+':
            last_cds_a = cds_a[-1]
            first_cds_b = cds_b[0]
        else:
            last_cds_a = cds_a[0]
            first_cds_b = cds_b[-1]

        # Calculate cumulative CDS length up to join point
        total_cds_a = sum(c.length for c in cds_a)
        remaining_a = total_cds_a % 3

        # Check if the first CDS of B starts in compatible frame
        if first_cds_b.phase in ('0', '1', '2'):
            expected_phase = remaining_a
            actual_phase = int(first_cds_b.phase)
            return expected_phase == actual_phase

        return True  # If phase unknown, don't block merge

    def merge_genes(self, gene_a: Gene, gene_b: Gene) -> Gene:
        """Merge two genes into a single gene model."""
        new_start = min(gene_a.start, gene_b.start)
        new_end = max(gene_a.end, gene_b.end)
        # Keep merge IDs short to avoid unbounded growth
        a_base = gene_a.attributes.get('merged_from', gene_a.gene_id)
        b_base = gene_b.gene_id
        merge_trail = f"{a_base},{b_base}"
        new_id = f"merged_{hash(merge_trail) & 0xFFFFFFFF:08x}"

        merged = Gene(
            gene_id=new_id,
            seqid=gene_a.seqid,
            strand=gene_a.strand,
            start=new_start,
            end=new_end,
            source='Refined',
            attributes={'ID': new_id,
                       'merged_from': merge_trail}
        )

        # Merge transcripts - combine exons and CDS from both, deduplicate
        if gene_a.transcripts and gene_b.transcripts:
            tx_a = gene_a.transcripts[0]
            tx_b = gene_b.transcripts[0]

            merged_tx = Transcript(
                transcript_id=f"{new_id}.1",
                seqid=gene_a.seqid,
                strand=gene_a.strand,
                start=new_start,
                end=new_end,
                source='Refined'
            )
            merged_tx.exons = self._deduplicate_features(tx_a.exons + tx_b.exons)
            merged_tx.cds = self._deduplicate_features(tx_a.cds + tx_b.cds)
            merged_tx.five_prime_utrs = self._deduplicate_features(
                tx_a.five_prime_utrs + tx_b.five_prime_utrs)
            merged_tx.three_prime_utrs = self._deduplicate_features(
                tx_a.three_prime_utrs + tx_b.three_prime_utrs)
            merged.transcripts.append(merged_tx)

        return merged

    @staticmethod
    def _deduplicate_features(features: List[Feature], tolerance: int = 5) -> List[Feature]:
        """Remove duplicate/near-duplicate features, keeping the longest."""
        if not features:
            return features
        # Sort by start position
        sorted_feats = sorted(features, key=lambda f: (f.start, f.end))
        result = [sorted_feats[0]]
        for feat in sorted_feats[1:]:
            prev = result[-1]
            # If nearly identical coordinates, skip
            if (abs(feat.start - prev.start) <= tolerance and
                abs(feat.end - prev.end) <= tolerance):
                # Keep the longer one
                if feat.length > prev.length:
                    result[-1] = feat
            else:
                result.append(feat)
        return result


# ============================================================================
# UTR recovery
# ============================================================================
class UTRRecovery:
    """Recover UTRs that may have been stripped by TransDecoder."""

    def __init__(self, genome: GenomeAccess, coverage: CoverageAccess,
                 bam_evidence=None):
        self.genome = genome
        self.coverage = coverage
        self.bam = bam_evidence or NoBAMEvidence()

    def recover_utrs(self, refined_gene: Gene, stringtie_genes: List[Gene]) -> Gene:
        """Check if UTRs from StringTie should be added to gene models."""
        if not refined_gene.transcripts:
            return refined_gene

        tx = refined_gene.transcripts[0]
        if not tx.cds:
            return refined_gene

        # Find overlapping StringTie transcripts
        overlapping_st = [
            sg for sg in stringtie_genes
            if (sg.seqid == refined_gene.seqid and
                sg.strand == refined_gene.strand and
                sg.start <= refined_gene.end and
                sg.end >= refined_gene.start)
        ]

        if not overlapping_st:
            return refined_gene

        sorted_cds = sorted(tx.cds, key=lambda c: c.start)

        for st_gene in overlapping_st:
            for st_tx in st_gene.transcripts:
                st_exons = st_tx.sorted_exons()
                if not st_exons:
                    continue

                # Check for 5' UTR recovery
                if refined_gene.strand == '+':
                    first_cds = sorted_cds[0]
                    # Look for StringTie exons upstream of first CDS.
                    # Require junction support on the intron connecting the
                    # candidate UTR exon to its downstream neighbor, so that
                    # unsupported terminal exons (e.g. from a low-quality
                    # StringTie transcript) are not re-introduced here after
                    # being correctly dropped by the terminal exon audit.
                    tx_exons_sorted = sorted(tx.exons, key=lambda e: e.start)
                    for exon in st_exons:
                        if exon.end < first_cds.start and exon.end >= refined_gene.start - 5000:
                            cov = self.coverage.get_mean_coverage(
                                refined_gene.seqid, exon.start, exon.end)
                            if cov > 2.0:
                                # Find nearest downstream exon already in the model
                                downstream = next(
                                    (e for e in tx_exons_sorted if e.start > exon.end),
                                    None)
                                if downstream is not None:
                                    intron_s = exon.end + 1
                                    intron_e = downstream.start - 1
                                    if (intron_e > intron_s and
                                            self.bam.count_spliced_reads(
                                                refined_gene.seqid,
                                                intron_s, intron_e) < 1):
                                        logger.debug(
                                            f"  UTR recovery: skipping 5'-exon "
                                            f"{exon.start}-{exon.end} — no junction "
                                            f"support on intron {intron_s}-{intron_e}")
                                        continue
                                utr = Feature(
                                    seqid=refined_gene.seqid, source='Refined',
                                    ftype='five_prime_UTR', start=exon.start,
                                    end=exon.end, score=0.0, strand=refined_gene.strand,
                                    phase='.', attributes={}
                                )
                                tx.five_prime_utrs.append(utr)
                                # Also add as exon if not already present
                                if not any(abs(e.start - exon.start) < 5 for e in tx.exons):
                                    new_exon = Feature(
                                        seqid=refined_gene.seqid, source='Refined',
                                        ftype='exon', start=exon.start, end=exon.end,
                                        score=0.0, strand=refined_gene.strand,
                                        phase='.', attributes={}
                                    )
                                    tx.exons.append(new_exon)
                                    refined_gene.start = min(refined_gene.start, exon.start)
                                    tx.start = min(tx.start, exon.start)

                    # Check for 3' UTR
                    last_cds = sorted_cds[-1]
                    for exon in st_exons:
                        if exon.start > last_cds.end and exon.start <= refined_gene.end + 5000:
                            cov = self.coverage.get_mean_coverage(
                                refined_gene.seqid, exon.start, exon.end)
                            if cov > 2.0:
                                utr = Feature(
                                    seqid=refined_gene.seqid, source='Refined',
                                    ftype='three_prime_UTR', start=exon.start,
                                    end=exon.end, score=0.0, strand=refined_gene.strand,
                                    phase='.', attributes={}
                                )
                                tx.three_prime_utrs.append(utr)
                                if not any(abs(e.start - exon.start) < 5 for e in tx.exons):
                                    new_exon = Feature(
                                        seqid=refined_gene.seqid, source='Refined',
                                        ftype='exon', start=exon.start, end=exon.end,
                                        score=0.0, strand=refined_gene.strand,
                                        phase='.', attributes={}
                                    )
                                    tx.exons.append(new_exon)
                                    refined_gene.end = max(refined_gene.end, exon.end)
                                    tx.end = max(tx.end, exon.end)

                else:  # minus strand
                    # For minus strand, 5' UTR is at higher coordinates
                    first_cds = sorted_cds[-1]  # highest coord CDS = 5' end
                    tx_exons_sorted = sorted(tx.exons, key=lambda e: e.start)
                    for exon in st_exons:
                        if exon.start > first_cds.end and exon.start <= refined_gene.end + 5000:
                            cov = self.coverage.get_mean_coverage(
                                refined_gene.seqid, exon.start, exon.end)
                            if cov > 2.0:
                                # On minus strand the 5'-terminal exon is at the
                                # high-coordinate end.  The inner intron connects
                                # this candidate to its upstream (lower-coord) neighbor.
                                upstream = next(
                                    (e for e in reversed(tx_exons_sorted)
                                     if e.end < exon.start), None)
                                if upstream is not None:
                                    intron_s = upstream.end + 1
                                    intron_e = exon.start - 1
                                    if (intron_e > intron_s and
                                            self.bam.count_spliced_reads(
                                                refined_gene.seqid,
                                                intron_s, intron_e) < 1):
                                        logger.debug(
                                            f"  UTR recovery: skipping 5'-exon "
                                            f"{exon.start}-{exon.end} (minus) — "
                                            f"no junction on intron {intron_s}-{intron_e}")
                                        continue
                                utr = Feature(
                                    seqid=refined_gene.seqid, source='Refined',
                                    ftype='five_prime_UTR', start=exon.start,
                                    end=exon.end, score=0.0, strand=refined_gene.strand,
                                    phase='.', attributes={}
                                )
                                tx.five_prime_utrs.append(utr)
                                if not any(abs(e.start - exon.start) < 5 for e in tx.exons):
                                    new_exon = Feature(
                                        seqid=refined_gene.seqid, source='Refined',
                                        ftype='exon', start=exon.start, end=exon.end,
                                        score=0.0, strand=refined_gene.strand,
                                        phase='.', attributes={}
                                    )
                                    tx.exons.append(new_exon)
                                    refined_gene.end = max(refined_gene.end, exon.end)
                                    tx.end = max(tx.end, exon.end)

                    last_cds = sorted_cds[0]  # lowest coord CDS = 3' end
                    for exon in st_exons:
                        if exon.end < last_cds.start and exon.end >= refined_gene.start - 5000:
                            cov = self.coverage.get_mean_coverage(
                                refined_gene.seqid, exon.start, exon.end)
                            if cov > 2.0:
                                utr = Feature(
                                    seqid=refined_gene.seqid, source='Refined',
                                    ftype='three_prime_UTR', start=exon.start,
                                    end=exon.end, score=0.0, strand=refined_gene.strand,
                                    phase='.', attributes={}
                                )
                                tx.three_prime_utrs.append(utr)
                                if not any(abs(e.start - exon.start) < 5 for e in tx.exons):
                                    new_exon = Feature(
                                        seqid=refined_gene.seqid, source='Refined',
                                        ftype='exon', start=exon.start, end=exon.end,
                                        score=0.0, strand=refined_gene.strand,
                                        phase='.', attributes={}
                                    )
                                    tx.exons.append(new_exon)
                                    refined_gene.start = min(refined_gene.start, exon.start)
                                    tx.start = min(tx.start, exon.start)

        return refined_gene


# ============================================================================
# ncRNA detection
# ============================================================================
class ncRNADetector:
    """Identify potential non-coding RNA genes."""

    def __init__(self, coverage: CoverageAccess):
        self.coverage = coverage

    def find_ncrna_candidates(self, seqid: str, stringtie_genes: List[Gene],
                               helixer_genes: List[Gene],
                               td_genes: List[Gene],
                               coding_regions: List[Tuple[int, int, str]]) -> List[Gene]:
        """Find StringTie transcripts with RNA-seq support but no coding potential."""
        ncrna_genes = []

        for st_gene in stringtie_genes:
            if st_gene.seqid != seqid:
                continue

            # Check if this overlaps any coding prediction (strand-agnostic for
            # unstranded transcripts, strand-specific for stranded)
            overlaps_coding = False
            for cstart, cend, cstrand in coding_regions:
                if st_gene.start <= cend and st_gene.end >= cstart:
                    # If StringTie has strand info, require strand match
                    if st_gene.strand in ('+', '-') and cstrand in ('+', '-'):
                        if st_gene.strand == cstrand:
                            overlaps_coding = True
                            break
                    else:
                        # Unstranded transcript - check position overlap with any coding gene
                        overlaps_coding = True
                        break

            if overlaps_coding:
                continue

            # Check for overlapping TransDecoder predictions
            overlaps_td = any(
                td.seqid == seqid and td.strand == st_gene.strand and
                td.start <= st_gene.end and td.end >= st_gene.start
                for td in td_genes
            )
            if overlaps_td:
                continue

            # Verify RNA-seq support
            if not st_gene.transcripts:
                continue

            tx = st_gene.transcripts[0]
            if not tx.exons:
                continue

            # Check coverage
            total_cov = 0
            exon_bp = 0
            for exon in tx.exons:
                cov = self.coverage.get_mean_coverage(seqid, exon.start, exon.end)
                total_cov += cov * exon.length
                exon_bp += exon.length

            if exon_bp == 0:
                continue

            mean_cov = total_cov / exon_bp

            # Require reasonable coverage for ncRNA call
            if mean_cov < 3.0:
                continue

            # Check FPKM if available
            fpkm = 0.0
            for stx in st_gene.transcripts:
                try:
                    fpkm = max(fpkm, float(stx.attributes.get('FPKM', '0')))
                except ValueError:
                    pass

            if fpkm < 0.5 and mean_cov < 5.0:
                continue

            # Create ncRNA gene
            ncrna = Gene(
                gene_id=f"ncRNA_{st_gene.gene_id}",
                seqid=seqid,
                strand=st_gene.strand,
                start=st_gene.start,
                end=st_gene.end,
                source='Refined',
                gene_type='ncRNA',
                attributes={
                    'ID': f"ncRNA_{st_gene.gene_id}",
                    'gene_biotype': 'ncRNA',
                    'mean_coverage': f"{mean_cov:.1f}",
                    'FPKM': f"{fpkm:.2f}"
                }
            )

            ncrna_tx = Transcript(
                transcript_id=f"ncRNA_{st_gene.gene_id}.1",
                seqid=seqid,
                strand=st_gene.strand,
                start=st_gene.start,
                end=st_gene.end,
                source='Refined'
            )
            ncrna_tx.exons = [Feature(
                seqid=e.seqid, source='Refined', ftype='exon',
                start=e.start, end=e.end, score=e.score,
                strand=e.strand, phase='.', attributes={}
            ) for e in tx.exons]
            ncrna.transcripts.append(ncrna_tx)
            ncrna_genes.append(ncrna)

        return ncrna_genes


# ============================================================================
# BAM spliced-read evidence
# ============================================================================
class SplicedReadEvidence:
    """Query BAM file for spliced RNA-seq reads supporting exon junctions."""

    def __init__(self, bam_path: str):
        import pysam

        # Detect file format: SAM files are text, BAM files start with magic bytes
        is_sam = False
        try:
            with open(bam_path, 'rb') as fh:
                magic = fh.read(4)
                # BAM magic: \x1f\x8b (gzip) or b'BAM\1'
                if magic[:2] != b'\x1f\x8b' and magic[:3] != b'BAM':
                    is_sam = True
        except Exception:
            pass

        if is_sam:
            # Convert SAM to BAM in-place (or to temp file)
            bam_out = bam_path.rsplit('.', 1)[0] + '.bam'
            logger.warning(f"Input file '{bam_path}' appears to be SAM format, "
                          f"not BAM. Converting to BAM...")
            try:
                import subprocess
                # Check if samtools is available
                subprocess.run(['samtools', 'view', '-bS', '-o', bam_out, bam_path],
                              check=True, capture_output=True)
                subprocess.run(['samtools', 'index', bam_out],
                              check=True, capture_output=True)
                logger.info(f"  Converted to: {bam_out}")
                bam_path = bam_out
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"Cannot convert SAM to BAM. Please convert manually:\n"
                            f"  samtools view -bS {bam_path} | samtools sort -o {bam_out}\n"
                            f"  samtools index {bam_out}\n"
                            f"Then rerun with --bam {bam_out}")
                self.available = False
                self.bam = None
                self.bam_refs = set()
                self._chrom_map = {}
                return

        try:
            self.bam = pysam.AlignmentFile(bam_path, "rb")
        except ValueError:
            self.bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)

        # Check for index
        has_idx = False
        try:
            has_idx = self.bam.has_index()
        except Exception:
            pass
        if not has_idx:
            logger.warning(f"BAM file has no index. Attempting to create one...")
            try:
                import subprocess
                subprocess.run(['samtools', 'index', bam_path],
                              check=True, capture_output=True)
                # Reopen with index
                self.bam.close()
                self.bam = pysam.AlignmentFile(bam_path, "rb")
                logger.info(f"  Index created successfully")
            except Exception:
                logger.warning(f"  Could not create index. Run: samtools index {bam_path}")

        self.available = True
        logger.info(f"BAM file loaded: {bam_path}")

        # Build chromosome name map for resolving mismatches
        try:
            self.bam_refs = set(self.bam.references)
        except Exception:
            self.bam_refs = set()
        self._chrom_map = {}

        try:
            logger.info(f"  References: {self.bam.nreferences}, "
                        f"Mapped reads: {self.bam.mapped}")
        except (ValueError, AttributeError):
            logger.info(f"  BAM references: {list(self.bam_refs)[:5]}...")

    def _resolve_chrom(self, seqid: str) -> str:
        """Resolve a GFF seqid to a BAM reference name."""
        if seqid in self._chrom_map:
            return self._chrom_map[seqid]

        # Direct match
        if seqid in self.bam_refs:
            self._chrom_map[seqid] = seqid
            return seqid

        # Try common transformations
        candidates = [
            seqid.split('|')[-1],           # lcl|scaffold_1 -> scaffold_1
            seqid.replace('lcl|', ''),       # lcl|scaffold_1 -> scaffold_1
            'lcl|' + seqid,                  # scaffold_1 -> lcl|scaffold_1
            'chr' + seqid,
            seqid.replace('chr', ''),
        ]
        for c in candidates:
            if c in self.bam_refs:
                self._chrom_map[seqid] = c
                logger.info(f"  BAM chrom resolved: {seqid} -> {c}")
                return c

        # No match found
        if not self._chrom_map.get('_warned_' + seqid):
            logger.warning(f"  BAM: seqid '{seqid}' not found in BAM references "
                          f"{list(self.bam_refs)[:3]}")
            self._chrom_map['_warned_' + seqid] = True
        self._chrom_map[seqid] = None
        return None

    def count_spliced_reads(self, seqid: str, intron_start: int, intron_end: int,
                            tolerance: int = 5) -> int:
        """Count reads with a splice junction matching this intron."""
        bam_chrom = self._resolve_chrom(seqid)
        if bam_chrom is None:
            return 0

        count = 0
        try:
            for read in self.bam.fetch(bam_chrom, intron_start - 1, intron_end):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if not read.cigartuples:
                    continue
                ref_pos = read.reference_start  # 0-based
                for op, length in read.cigartuples:
                    if op == 3:  # BAM_CREF_SKIP (N in CIGAR = intron)
                        skip_start = ref_pos + 1       # convert to 1-based
                        skip_end = ref_pos + length     # 1-based inclusive end
                        if (abs(skip_start - intron_start) <= tolerance and
                                abs(skip_end - intron_end) <= tolerance):
                            count += 1
                            break
                    if op in (0, 2, 3, 7, 8):  # M, D, N, =, X
                        ref_pos += length
        except (ValueError, KeyError):
            pass
        return count

    def has_junction_support(self, seqid: str, intron_start: int, intron_end: int,
                             min_reads: int = 2) -> bool:
        """Check if at least min_reads support this splice junction."""
        return self.count_spliced_reads(seqid, intron_start, intron_end) >= min_reads

    def reads_at_donor(self, seqid: str, donor_pos: int, tolerance: int = 5) -> int:
        """Max read count for any junction starting at donor_pos+1 (±tolerance)."""
        intron_start = donor_pos + 1
        best = 0
        for js, _, jc in self.find_novel_junctions(
                seqid, donor_pos - tolerance, donor_pos + tolerance + 1, min_reads=1):
            if abs(js - intron_start) <= tolerance:
                best = max(best, jc)
        return best

    def reads_at_acceptor(self, seqid: str, acceptor_pos: int, tolerance: int = 5) -> int:
        """Max read count for any junction ending at acceptor_pos-1 (±tolerance)."""
        intron_end = acceptor_pos - 1
        best = 0
        for _, je, jc in self.find_novel_junctions(
                seqid, acceptor_pos - 200000, acceptor_pos + tolerance, min_reads=1):
            if abs(je - intron_end) <= tolerance:
                best = max(best, jc)
        return best

    def find_novel_junctions(self, seqid: str, start: int, end: int,
                              min_reads: int = 3) -> List[Tuple[int, int, int]]:
        """Find all splice junctions in a region supported by at least min_reads.

        Returns list of (intron_start, intron_end, read_count) tuples.
        Useful for discovering junctions not predicted by Helixer/StringTie.
        """
        bam_chrom = self._resolve_chrom(seqid)
        if bam_chrom is None:
            return []

        junction_counts = {}
        try:
            for read in self.bam.fetch(bam_chrom, start - 1, end):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if not read.cigartuples:
                    continue
                ref_pos = read.reference_start
                for op, length in read.cigartuples:
                    if op == 3:
                        j_start = ref_pos + 1
                        j_end = ref_pos + length
                        if start <= j_start and j_end <= end:
                            key = (j_start, j_end)
                            junction_counts[key] = junction_counts.get(key, 0) + 1
                    if op in (0, 2, 3, 7, 8):
                        ref_pos += length
        except (ValueError, KeyError):
            pass

        return [(s, e, c) for (s, e), c in junction_counts.items() if c >= min_reads]

    def get_junction_read_count_for_gene(self, gene: 'Gene') -> Dict[Tuple[int, int], int]:
        """Get spliced read counts for all introns in a gene."""
        result = {}
        if not gene.transcripts:
            return result
        tx = gene.transcripts[0]
        for intron_start, intron_end in tx.introns():
            count = self.count_spliced_reads(gene.seqid, intron_start, intron_end)
            result[(intron_start, intron_end)] = count
        return result

    def close(self):
        self.bam.close()


class MultiSplicedReadEvidence:
    """Aggregate queries across multiple BAM files.

    Holds a list of SplicedReadEvidence handles and forwards each method,
    summing read counts (count_spliced_reads, reads_at_donor,
    reads_at_acceptor) or merging junction lists (find_novel_junctions,
    find_junctions_starting_in, find_junctions_ending_in).
    """

    def __init__(self, paths):
        self._handles = [SplicedReadEvidence(p) for p in paths]
        self._handles = [h for h in self._handles if getattr(h, 'available', False)]
        self.available = bool(self._handles)
        if self.available:
            logger.info(f"  MultiSplicedReadEvidence: aggregating "
                        f"{len(self._handles)} BAM file(s)")

    # Sum-style queries
    def count_spliced_reads(self, *args, **kwargs) -> int:
        return sum(h.count_spliced_reads(*args, **kwargs) for h in self._handles)

    def reads_at_donor(self, *args, **kwargs) -> int:
        return sum(h.reads_at_donor(*args, **kwargs) for h in self._handles)

    def reads_at_acceptor(self, *args, **kwargs) -> int:
        return sum(h.reads_at_acceptor(*args, **kwargs) for h in self._handles)

    def has_junction_support(self, *args, **kwargs) -> bool:
        return self.count_spliced_reads(*args, **kwargs) >= kwargs.get('min_reads', 2)

    # Merge-style queries: concatenate per-file (j_start, j_end, count) tuples,
    # then collapse identical (j_start, j_end) by summing counts.
    def _merge_junctions(self, lists):
        merged = {}
        for lst in lists:
            for js, je, jc in lst:
                merged[(js, je)] = merged.get((js, je), 0) + jc
        return [(js, je, jc) for (js, je), jc in merged.items()]

    def find_novel_junctions(self, *args, **kwargs):
        return self._merge_junctions(
            [h.find_novel_junctions(*args, **kwargs) for h in self._handles])

    def find_junctions_starting_in(self, *args, **kwargs):
        return self._merge_junctions(
            [h.find_junctions_starting_in(*args, **kwargs)
             for h in self._handles
             if hasattr(h, 'find_junctions_starting_in')])

    def find_junctions_ending_in(self, *args, **kwargs):
        return self._merge_junctions(
            [h.find_junctions_ending_in(*args, **kwargs)
             for h in self._handles
             if hasattr(h, 'find_junctions_ending_in')])

    def get_intron_support(self, gene):
        result = {}
        if not gene.transcripts:
            return result
        for intron_start, intron_end in gene.transcripts[0].introns():
            result[(intron_start, intron_end)] = self.count_spliced_reads(
                gene.seqid, intron_start, intron_end)
        return result

    def close(self):
        for h in self._handles:
            try:
                h.close()
            except Exception:
                pass


class JunctionFileEvidence:
    """Pre-computed splice junction evidence from Portcullis or similar tools.

    Replaces BAM scanning with an in-memory lookup table loaded once at startup.
    This is dramatically faster than scanning a BAM file per-intron, since all
    junctions are loaded into a dict keyed by (seqid, start, end).

    Accepted formats (auto-detected):
    - Portcullis tab (*.tab): columns include chrom, start, end, strand, raw/score
    - Portcullis BED (*.bed): BED6+ with score = read count, thickStart/End = intron
    - STAR SJ.out.tab: cols = chrom, start, end, strand, motif, annotated, unique, multi, overhang
    - Generic tab/BED: any tab file with chrom, start, end as first 3 columns
      and an integer read count in columns 5+ (auto-detected)
    """

    def __init__(self, junction_path):
        self.available = True
        # Dict: (seqid, intron_start, intron_end) -> read_count (summed across files)
        self._junctions = {}
        # Interval index built from the merged dict (so per-junction reads are
        # counted once even when present in multiple files).
        self._by_seqid = defaultdict(list)

        if isinstance(junction_path, str):
            paths = [junction_path]
        else:
            paths = list(junction_path)

        total = 0
        for p in paths:
            total += self._load(p)
            logger.info(f"Junction file loaded: {p}")

        # Build sorted index from merged dict
        self._by_seqid.clear()
        for (seqid, js, je), count in self._junctions.items():
            self._by_seqid[seqid].append((js, je, count))
        for seqid in self._by_seqid:
            self._by_seqid[seqid].sort()

        logger.info(f"  {len(self._junctions)} unique splice junctions loaded "
                    f"({total} records read across {len(paths)} file(s))")
        seqids = sorted(self._by_seqid.keys())
        logger.info(f"  Sequences: {len(seqids)} ({', '.join(seqids[:5])}{'...' if len(seqids) > 5 else ''})")

    def _load(self, path: str) -> int:
        """Auto-detect format and load junctions.

        Supported formats:
        - Portcullis BED12: thickStart/thickEnd = intron coords (columns 7-8)
        - STAR SJ.out.tab: chrom, start, end, strand, motif, annotated, unique, multi, overhang
        - Portcullis tab: header with 'index', 'refname', 'start', 'end', 'nb_raw_aln'
        - Generic BED/tab: chrom, start, end with optional count
        """
        n = 0
        with open(path) as f:
            # Peek at first lines for format detection
            header_line = None
            first_data = None

            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if stripped.startswith('track '):
                    header_line = stripped
                    continue
                parts = stripped.split('\t')
                # Portcullis tab header: first field is "index"
                if parts[0] == 'index':
                    header_line = stripped
                    continue
                first_data = stripped
                break

            if not first_data:
                logger.warning(f"Junction file has no data lines: {path}")
                return 0

            parts = first_data.split('\t')
            n_cols = len(parts)

            # Detect format
            fmt = 'unknown'

            if header_line and header_line.startswith('track '):
                # BED file with track line
                if n_cols >= 12:
                    fmt = 'portcullis_bed12'
                else:
                    fmt = 'bed'
            elif header_line and header_line.split('\t')[0].strip().lower() == 'index':
                fmt = 'portcullis_tab'
                header_fields = header_line.strip().split('\t')
                self._col_map = {h.strip().lower(): i
                                 for i, h in enumerate(header_fields)}
            elif n_cols == 9:
                try:
                    int(parts[1]); int(parts[2]); int(parts[6]); int(parts[7])
                    fmt = 'star'
                except ValueError:
                    pass
            elif n_cols >= 6 and not parts[0].isdigit():
                try:
                    int(parts[1]); int(parts[2])
                    fmt = 'bed'
                except ValueError:
                    pass

            if fmt == 'unknown':
                fmt = 'generic'

            logger.info(f"  Junction file format detected: {fmt} ({n_cols} columns)")

            # Re-read from beginning
            f.seek(0)
            for line in f:
                stripped = line.strip()
                if (not stripped or stripped.startswith('#')
                        or stripped.startswith('track ')):
                    continue
                parts = stripped.split('\t')
                if len(parts) < 3:
                    continue
                if parts[0] == 'index':
                    continue

                try:
                    if fmt == 'portcullis_bed12':
                        # Portcullis BED12:
                        # col 0=chrom, 1=chromStart(0-based), 2=chromEnd
                        # col 6=thickStart(0-based) = intron start
                        # col 7=thickEnd = intron end (0-based exclusive)
                        seqid = parts[0]
                        intron_start = int(parts[6]) + 1  # 0-based to 1-based
                        intron_end = int(parts[7])         # exclusive = 1-based inclusive
                        # Portcullis BED score is confidence, not read count.
                        # All passing junctions are reliable; use count=1.
                        count = 1

                    elif fmt == 'portcullis_tab':
                        col = getattr(self, '_col_map', {})
                        idx_refname = col.get('refname', 2)
                        idx_start = col.get('start', 4)
                        idx_end = col.get('end', 5)
                        idx_raw = col.get('nb_raw_aln', 19)

                        seqid = parts[idx_refname]
                        # Portcullis tab uses 0-based coordinates for both start and end:
                        # start = 0-based last position of upstream exon (= intron_start - 1)
                        # end   = 0-based last position of intron       (= exon_b.start - 1)
                        # Both need +1 to convert to 1-based pipeline coordinates.
                        intron_start = int(parts[idx_start]) + 1
                        intron_end = int(parts[idx_end]) + 1
                        try:
                            count = int(parts[idx_raw])
                        except (ValueError, IndexError):
                            count = 1

                    elif fmt == 'star':
                        seqid = parts[0]
                        intron_start = int(parts[1])
                        intron_end = int(parts[2])
                        count = int(parts[6]) + int(parts[7])

                    elif fmt == 'bed':
                        seqid = parts[0]
                        intron_start = int(parts[1]) + 1
                        intron_end = int(parts[2])
                        count = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 1

                    else:
                        seqid = parts[0]
                        intron_start = int(parts[1])
                        intron_end = int(parts[2])
                        count = 1
                        for ci in range(3, min(9, len(parts))):
                            if parts[ci].isdigit() and int(parts[ci]) > 0:
                                count = int(parts[ci])
                                break

                    if intron_end > intron_start and count > 0:
                        key = (seqid, intron_start, intron_end)
                        self._junctions[key] = self._junctions.get(key, 0) + count
                        self._by_seqid[seqid].append((intron_start, intron_end, count))
                        n += 1

                except (ValueError, IndexError):
                    continue

        return n

    def count_spliced_reads(self, seqid: str, intron_start: int, intron_end: int,
                            tolerance: int = 5) -> int:
        """Look up junction support from pre-computed table.
        O(1) for exact matches, O(tolerance) for fuzzy matches."""
        # Try exact match first
        exact = self._junctions.get((seqid, intron_start, intron_end))
        if exact is not None:
            return exact

        # Fuzzy match within tolerance
        if tolerance > 0:
            best = 0
            for ds in range(-tolerance, tolerance + 1):
                for de in range(-tolerance, tolerance + 1):
                    count = self._junctions.get(
                        (seqid, intron_start + ds, intron_end + de), 0)
                    if count > best:
                        best = count
            return best

        return 0

    def has_junction_support(self, seqid: str, intron_start: int, intron_end: int,
                             min_reads: int = 2) -> bool:
        return self.count_spliced_reads(seqid, intron_start, intron_end) >= min_reads

    def reads_at_donor(self, seqid: str, donor_pos: int, tolerance: int = 5) -> int:
        """Max read count for any junction whose intron starts at donor_pos+1 (±tolerance)."""
        import bisect
        junctions_on_seq = self._by_seqid.get(seqid, [])
        if not junctions_on_seq:
            return 0
        intron_start = donor_pos + 1
        idx = bisect.bisect_left(junctions_on_seq, (intron_start - tolerance, 0, 0))
        best = 0
        while idx < len(junctions_on_seq):
            j_start, _, j_count = junctions_on_seq[idx]
            if j_start > intron_start + tolerance:
                break
            if abs(j_start - intron_start) <= tolerance:
                best = max(best, j_count)
            idx += 1
        return best

    def reads_at_acceptor(self, seqid: str, acceptor_pos: int, tolerance: int = 5) -> int:
        """Max read count for any junction whose intron ends at acceptor_pos-1 (±tolerance)."""
        import bisect
        junctions_on_seq = self._by_seqid.get(seqid, [])
        if not junctions_on_seq:
            return 0
        intron_end = acceptor_pos - 1
        # Scan junctions starting up to ~200 kb before to find any ending here
        idx = bisect.bisect_left(junctions_on_seq, (intron_end - 200000, 0, 0))
        best = 0
        while idx < len(junctions_on_seq):
            j_start, j_end, j_count = junctions_on_seq[idx]
            if j_start > intron_end + tolerance:
                break
            if abs(j_end - intron_end) <= tolerance:
                best = max(best, j_count)
            idx += 1
        return best

    def find_junctions_starting_in(self, seqid: str, start_lower: int, start_upper: int,
                                    min_reads: int = 2) -> List[Tuple[int, int, int]]:
        """Find all junctions whose intron START falls within [start_lower, start_upper].

        Unlike find_novel_junctions, does NOT restrict the intron end, so junctions
        that span beyond the search window are still returned.
        """
        import bisect
        junctions_on_seq = self._by_seqid.get(seqid, [])
        if not junctions_on_seq:
            return []
        idx = bisect.bisect_left(junctions_on_seq, (start_lower, 0, 0))
        results = []
        while idx < len(junctions_on_seq):
            j_start, j_end, j_count = junctions_on_seq[idx]
            if j_start > start_upper:
                break
            if j_count >= min_reads:
                results.append((j_start, j_end, j_count))
            idx += 1
        return results

    def find_junctions_ending_in(self, seqid: str, end_lower: int, end_upper: int,
                                  min_reads: int = 2) -> List[Tuple[int, int, int]]:
        """Find junctions whose intron END falls within [end_lower, end_upper].

        Used to detect introns landing inside a terminal exon — indicating the
        exon should be trimmed and additional downstream exons recovered.
        """
        import bisect
        junctions_on_seq = self._by_seqid.get(seqid, [])
        if not junctions_on_seq:
            return []
        idx = bisect.bisect_left(junctions_on_seq, (end_lower - 500_000, 0, 0))
        results = []
        while idx < len(junctions_on_seq):
            j_start, j_end, j_count = junctions_on_seq[idx]
            if j_start > end_upper:
                break
            if end_lower <= j_end <= end_upper and j_count >= min_reads:
                results.append((j_start, j_end, j_count))
            idx += 1
        return results

    def find_novel_junctions(self, seqid: str, start: int, end: int,
                              min_reads: int = 3) -> List[Tuple[int, int, int]]:
        """Find all junctions in a region using binary search on sorted list."""
        import bisect
        junctions_on_seq = self._by_seqid.get(seqid, [])
        if not junctions_on_seq:
            return []

        # Binary search for start of region
        idx = bisect.bisect_left(junctions_on_seq, (start, 0, 0))
        results = []
        while idx < len(junctions_on_seq):
            j_start, j_end, j_count = junctions_on_seq[idx]
            if j_start > end:
                break
            if j_end <= end and j_count >= min_reads:
                results.append((j_start, j_end, j_count))
            idx += 1
        return results

    def get_junction_read_count_for_gene(self, gene: 'Gene') -> Dict[Tuple[int, int], int]:
        """Get junction counts for all introns in a gene."""
        result = {}
        if not gene.transcripts:
            return result
        tx = gene.transcripts[0]
        for intron_start, intron_end in tx.introns():
            count = self.count_spliced_reads(gene.seqid, intron_start, intron_end)
            result[(intron_start, intron_end)] = count
        return result

    def close(self):
        pass


class NoBAMEvidence:
    """Stub class when no BAM file is provided."""

    def __init__(self):
        self.available = False

    def count_spliced_reads(self, *args, **kwargs):
        return 0

    def has_junction_support(self, *args, **kwargs):
        return False

    def reads_at_donor(self, *_args, **_kwargs):
        return 0

    def reads_at_acceptor(self, *_args, **_kwargs):
        return 0

    def find_novel_junctions(self, *args, **kwargs):
        return []

    def find_junctions_starting_in(self, *_args, **_kwargs):
        return []

    def find_junctions_ending_in(self, *_args, **_kwargs):
        return []

    def get_junction_read_count_for_gene(self, *args, **kwargs):
        return {}

    def close(self):
        pass


# ============================================================================
# Splice site re-evaluation (UTR vs intron)
# ============================================================================
class SpliceSiteEvaluator:
    """Evaluate whether poorly supported splice sites might actually be UTRs."""

    def __init__(self, genome: GenomeAccess, coverage: CoverageAccess,
                 bam_evidence=None):
        self.genome = genome
        self.coverage = coverage
        self.bam = bam_evidence or NoBAMEvidence()

    def evaluate_5prime_splice(self, gene: Gene) -> Gene:
        """Check if 5' splice sites are supported or if region is better modeled as UTR."""
        if not gene.transcripts:
            return gene

        tx = gene.transcripts[0]
        exons = tx.sorted_exons()
        if len(exons) < 2:
            return gene

        # Validate exon coordinates
        for ex in exons:
            if ex.start > ex.end or ex.start < 1:
                logger.warning(f"  Gene {gene.gene_id}: skipping 5' splice eval — "
                             f"malformed exon {ex.start}-{ex.end}")
                return gene

        # Check first intron (between exon 1 and exon 2) for 5' UTR possibility
        if gene.strand == '+':
            first_exon = exons[0]
            second_exon = exons[1]
        else:
            first_exon = exons[-1]
            second_exon = exons[-2]

        intron_start = first_exon.end + 1 if gene.strand == '+' else second_exon.end + 1
        intron_end = second_exon.start - 1 if gene.strand == '+' else first_exon.start - 1

        if intron_end <= intron_start:
            return gene

        # Get coverage in intron region
        intron_cov = self.coverage.get_mean_coverage(gene.seqid, intron_start, intron_end)
        exon1_cov = self.coverage.get_mean_coverage(gene.seqid, first_exon.start, first_exon.end)
        exon2_cov = self.coverage.get_mean_coverage(gene.seqid, second_exon.start, second_exon.end)

        # If intron has substantial coverage relative to exons, might be UTR
        avg_exon_cov = (exon1_cov + exon2_cov) / 2.0
        if avg_exon_cov < 1.0:
            return gene

        intron_ratio = intron_cov / avg_exon_cov

        # Check splice site quality.
        # Use intron_start/intron_end (already strand-correct) rather than
        # first_exon.end / second_exon.start, which point to outer exon
        # boundaries for minus-strand genes and are not adjacent to the intron.
        donor_seq = self.genome.get_splice_donor(gene.seqid, intron_start - 1, gene.strand)
        acceptor_seq = self.genome.get_splice_acceptor(gene.seqid, intron_end + 1, gene.strand)

        donor_s = score_donor(donor_seq) if len(donor_seq) == DONOR_LEN else -5.0
        acceptor_s = score_acceptor(acceptor_seq) if len(acceptor_seq) == ACCEPTOR_LEN else -5.0

        # If coverage doesn't drop much AND splice sites are weak, convert to UTR
        # But first check BAM for direct spliced-read evidence
        bam_supports_intron = False
        if self.bam.available:
            n_spliced = self.bam.count_spliced_reads(
                gene.seqid, intron_start, intron_end)
            if n_spliced >= 2:
                bam_supports_intron = True
                logger.debug(f"Gene {gene.gene_id}: BAM has {n_spliced} spliced reads "
                           f"supporting first intron; keeping as intron")

        if (intron_ratio > 0.5 and donor_s < 1.0 and acceptor_s < 1.0
                and not bam_supports_intron):
            logger.info(f"Gene {gene.gene_id}: Converting first intron to UTR "
                       f"(intron_ratio={intron_ratio:.2f}, "
                       f"donor_score={donor_s:.1f}, acceptor_score={acceptor_s:.1f})")

            # Check if there's a start ATG in the second exon
            cds = tx.sorted_cds()
            if cds:
                if gene.strand == '+':
                    # Merge first exon into UTR, extending gene model
                    merged_exon = Feature(
                        seqid=gene.seqid, source='Refined', ftype='exon',
                        start=first_exon.start, end=second_exon.end,
                        score=0.0, strand=gene.strand, phase='.', attributes={}
                    )
                    # Replace first two exons with merged exon
                    tx.exons = [merged_exon] + exons[2:]
                    # Add UTR for the merged region up to CDS start
                    first_cds = cds[0]
                    if first_exon.start < first_cds.start:
                        utr = Feature(
                            seqid=gene.seqid, source='Refined',
                            ftype='five_prime_UTR',
                            start=first_exon.start, end=min(intron_end, first_cds.start - 1),
                            score=0.0, strand=gene.strand, phase='.', attributes={}
                        )
                        tx.five_prime_utrs.append(utr)

        return gene


# ============================================================================
# Main integration pipeline
# ============================================================================
class GeneAnnotationRefiner:
    """Main pipeline for integrating and refining gene annotations."""

    def __init__(self, genome_path: str, helixer_path: str = None,
                 stringtie_path: str = None, transdecoder_path: str = None,
                 bigwig_path: str = None, bam_path: str = None,
                 bigwig_fwd_path: str = None, bigwig_rev_path: str = None,
                 junctions_path: str = None,
                 manual_annotation_path: str = None,
                 refine_existing_path: str = None,
                 evidence_refinement: bool = True,
                 scoring_config: 'ScoringConfig' = None,
                 pwm_organism: str = 'drosophila',
                 tracer: 'GeneTracer' = None):
        self.genome = GenomeAccess(genome_path)
        self.coverage = CoverageAccess(bigwig_path) if bigwig_path else NoCoverageAccess()
        self.stranded_coverage = StrandedCoverage(bigwig_fwd_path, bigwig_rev_path)
        if self.stranded_coverage.available:
            logger.info("Stranded coverage available — will be used to veto "
                        "antisense-driven UTR extension and downstream-exon "
                        "recovery.")
        self.cfg = scoring_config or ScoringConfig()
        self.pwm_organism = pwm_organism
        self.tracer = tracer or GeneTracer()
        if self.tracer.enabled:
            tgt = (f"genes={self.tracer.gene_ids}" if self.tracer.gene_ids else "")
            rgs = (f"regions={self.tracer.regions}" if self.tracer.regions else "")
            logger.info(f"[TRACE] Gene tracer ENABLED — "
                        f"{' '.join(x for x in (tgt, rgs) if x)}")

        self.ncrna_threshold = self.cfg.ncrna_threshold
        self.coding_threshold = self.cfg.coding_threshold

        # Mode flag: refine-existing skips consensus building
        self.refine_existing_mode = refine_existing_path is not None
        # Whether to apply evidence-based refinement steps (junction validation,
        # coverage trimming, etc.). In refine-existing mode, default is to trust
        # the existing annotation and skip these steps unless explicitly requested.
        self.evidence_refinement = evidence_refinement

        # Parse input annotations (all optional)
        self.helixer_genes = []
        self.td_genes = []
        self.st_genes = []
        self.manual_genes = []
        self.existing_genes = []

        if helixer_path:
            logger.info("Parsing Helixer GFF...")
            self.helixer_genes = parse_helixer_gff(helixer_path)
            logger.info(f"  Found {len(self.helixer_genes)} Helixer gene models")

        if transdecoder_path:
            logger.info("Parsing TransDecoder GFF...")
            self.td_genes = parse_transdecoder_gff(transdecoder_path)
            logger.info(f"  Found {len(self.td_genes)} TransDecoder gene models")

        if stringtie_path:
            logger.info("Parsing StringTie GTF...")
            self.st_genes = parse_stringtie_gtf(stringtie_path)
            logger.info(f"  Found {len(self.st_genes)} StringTie transcripts")

        if manual_annotation_path:
            logger.info("Parsing manual annotation GFF...")
            self.manual_genes = parse_generic_gff3(manual_annotation_path,
                                                    source_label='Manual')
            logger.info(f"  Found {len(self.manual_genes)} manually annotated gene models")
            apply_manual_annotation_confidence(self.manual_genes)

        if refine_existing_path:
            logger.info("Parsing existing annotation for refinement...")
            self.existing_genes = parse_generic_gff3(refine_existing_path,
                                                      source_label='Existing')
            logger.info(f"  Found {len(self.existing_genes)} existing gene models")

        # Compute splice site PWMs from StringTie junctions (if available)
        global DONOR_PWM, ACCEPTOR_PWM
        if self.st_genes:
            logger.info("Computing splice site PWMs from StringTie junctions...")
            pwm_builder = SplicePWMBuilder(self.genome)
            DONOR_PWM, ACCEPTOR_PWM = pwm_builder.build_from_stringtie(
                self.st_genes, fallback_organism=self.pwm_organism)
        else:
            logger.info("No StringTie data; using fallback splice site PWMs")
            pwm_builder = SplicePWMBuilder(self.genome)
            DONOR_PWM, ACCEPTOR_PWM = pwm_builder._fallback_pwm(self.pwm_organism)

        # Load splice junction evidence (prefer pre-computed junctions over BAM).
        # Both inputs accept a single path or a list of paths.
        def _to_existing_list(p):
            if not p:
                return []
            paths = [p] if isinstance(p, str) else list(p)
            return [x for x in paths if x and os.path.exists(x)]

        junc_paths = _to_existing_list(junctions_path)
        bam_paths = _to_existing_list(bam_path)
        if junc_paths:
            logger.info(f"Loading pre-computed junctions: "
                        f"{len(junc_paths)} file(s)")
            self.bam_evidence = JunctionFileEvidence(junc_paths)
            if bam_paths:
                logger.info("  --junctions provided; --bam will be ignored for "
                           "junction evidence")
        elif bam_paths:
            logger.info(f"Loading BAM file(s): {len(bam_paths)}")
            if len(bam_paths) == 1:
                self.bam_evidence = SplicedReadEvidence(bam_paths[0])
            else:
                self.bam_evidence = MultiSplicedReadEvidence(bam_paths)
        else:
            if junctions_path:
                logger.warning(f"Junctions file(s) not found: {junctions_path}")
            if bam_path:
                logger.warning(f"BAM file(s) not found: {bam_path}; "
                              f"proceeding without BAM evidence")
            else:
                logger.info("No BAM or junctions file provided; splice junction "
                           "evidence from reads unavailable")
            self.bam_evidence = NoBAMEvidence()

        # Calibrate exon evidence distributions from StringTie data
        logger.info("Calibrating exon evidence distributions from StringTie...")
        self.calibrator = ExonEvidenceCalibrator(
            self.st_genes, self.coverage, self.bam_evidence)

        # Build spatial index for fast overlap queries
        logger.info("Building evidence spatial index...")
        self.evidence_index = EvidenceIndex()
        if self.helixer_genes:
            self.evidence_index.add_genes(self.helixer_genes, 'Helixer')
        if self.td_genes:
            self.evidence_index.add_genes(self.td_genes, 'TransDecoder')
        if self.st_genes:
            self.evidence_index.add_genes(self.st_genes, 'StringTie')
        self.evidence_index.build()

        # Initialize analysis modules
        self.posterior_calc = PosteriorCalculator(self.genome, self.coverage,
                                                  self.bam_evidence,
                                                  config=self.cfg,
                                                  evidence_index=self.evidence_index)
        self.merger = GeneMerger(self.genome, self.coverage, self.bam_evidence,
                                 st_genes=self.st_genes, tracer=self.tracer)
        self.utr_recovery = UTRRecovery(self.genome, self.coverage,
                                         self.bam_evidence)
        self.splice_eval = SpliceSiteEvaluator(self.genome, self.coverage,
                                                self.bam_evidence)
        self.ncrna_detector = ncRNADetector(self.coverage)

    def _incorporate_manual_genes(self, consensus_genes: List[Gene]) -> List[Gene]:
        """Incorporate manually annotated gene models into the consensus.

        Manual genes take priority: if a manual gene overlaps a consensus gene
        on the same strand, the consensus gene is replaced by the manual gene.
        Manual genes with no overlap are simply added.
        """
        logger.info(f"\nIncorporating {len(self.manual_genes)} manual annotations...")

        # Index consensus genes by position
        result = list(consensus_genes)
        replaced = set()

        for mg in self.manual_genes:
            # Find overlapping consensus genes on same strand
            overlapping_indices = []
            for i, cg in enumerate(result):
                if (i not in replaced and
                    cg.seqid == mg.seqid and cg.strand == mg.strand and
                    cg.start <= mg.end and cg.end >= mg.start):
                    # Require substantial overlap
                    overlap = min(cg.end, mg.end) - max(cg.start, mg.start) + 1
                    min_len = min(cg.end - cg.start + 1, mg.end - mg.start + 1)
                    if overlap > min_len * 0.3:
                        overlapping_indices.append(i)

            # Create the manual gene entry
            manual_gene = Gene(
                gene_id=mg.gene_id,
                seqid=mg.seqid,
                strand=mg.strand,
                start=mg.start,
                end=mg.end,
                source='Refined',
                attributes=dict(mg.attributes)
            )
            manual_gene.attributes['manual_annotation'] = 'true'
            manual_gene.attributes['evidence_sources'] = 'Manual'

            for mtx in mg.transcripts:
                ctx = Transcript(
                    transcript_id=mtx.transcript_id,
                    seqid=mtx.seqid,
                    strand=mtx.strand,
                    start=mtx.start,
                    end=mtx.end,
                    source='Refined'
                )
                ctx.exons = list(mtx.exons)
                ctx.cds = list(mtx.cds)
                ctx.five_prime_utrs = list(mtx.five_prime_utrs)
                ctx.three_prime_utrs = list(mtx.three_prime_utrs)
                manual_gene.transcripts.append(ctx)

            if overlapping_indices:
                # Replace the first overlapping gene, mark the rest for removal
                first_idx = overlapping_indices[0]
                result[first_idx] = manual_gene
                for idx in overlapping_indices[1:]:
                    replaced.add(idx)
                logger.info(f"  Manual gene {mg.gene_id} replaces "
                           f"{len(overlapping_indices)} consensus gene(s)")
            else:
                # No overlap — add as a new gene
                result.append(manual_gene)
                logger.info(f"  Manual gene {mg.gene_id} added (no overlap)")

        # Remove replaced genes
        if replaced:
            result = [g for i, g in enumerate(result) if i not in replaced]

        logger.info(f"  After incorporating manual annotations: "
                   f"{len(result)} total genes")
        return result

    def _validate_gene_coordinates(self, genes: List[Gene]) -> List[Gene]:
        """Validate gene/exon/CDS coordinates. Remove unfixable genes."""
        valid = []
        for gene in genes:
            problems = []

            # Check gene-level coords (catch inf, non-int, inverted, sub-1)
            if (not isinstance(gene.start, int) or not isinstance(gene.end, int)
                    or gene.start > gene.end or gene.start < 1):
                problems.append(f"gene coords {gene.start}-{gene.end}")

            # Check all transcript features
            for tx in gene.transcripts:
                for ex in tx.exons:
                    if ex.start > ex.end or ex.start < 1:
                        problems.append(f"exon {ex.start}-{ex.end}")
                for c in tx.cds:
                    if c.start > c.end or c.start < 1:
                        problems.append(f"CDS {c.start}-{c.end}")

            if problems:
                logger.warning(f"  Skipping {gene.gene_id} ({gene.seqid}:"
                             f"{gene.start}-{gene.end} {gene.strand}): "
                             f"malformed coordinates: {'; '.join(problems)}")
                continue

            # Ensure gene bounds encompass all features
            all_starts = [ex.start for tx in gene.transcripts for ex in tx.exons]
            all_ends = [ex.end for tx in gene.transcripts for ex in tx.exons]
            if all_starts and all_ends:
                gene.start = min(gene.start, min(all_starts))
                gene.end = max(gene.end, max(all_ends))

            valid.append(gene)
        return valid

    def refine(self) -> List[Gene]:
        """Run the full refinement pipeline."""
        logger.info("=" * 60)
        logger.info("Starting gene annotation refinement pipeline")
        if self.refine_existing_mode:
            logger.info("  Mode: REFINE EXISTING annotation")
        else:
            logger.info("  Mode: FULL CONSENSUS from prediction sources")
        if self.manual_genes:
            logger.info(f"  Manual annotations: {len(self.manual_genes)} genes")
        logger.info("=" * 60)

        if self.tracer.enabled:
            self.tracer.snapshot("Input: Helixer", self.helixer_genes)
            self.tracer.snapshot("Input: TransDecoder", self.td_genes)
            self.tracer.snapshot("Input: StringTie", self.st_genes)
            if self.existing_genes:
                self.tracer.snapshot("Input: Existing", self.existing_genes)
            if self.manual_genes:
                self.tracer.snapshot("Input: Manual", self.manual_genes)

        if self.refine_existing_mode:
            # --- Refine-existing mode ---
            # Start from the existing annotation; skip consensus building
            logger.info("\nStep 1 (refine mode): Loading existing annotation as base...")
            consensus_genes = []
            for eg in self.existing_genes:
                cgene = Gene(
                    gene_id=eg.gene_id,
                    seqid=eg.seqid,
                    strand=eg.strand,
                    start=eg.start,
                    end=eg.end,
                    source='Refined',
                    attributes=dict(eg.attributes)
                )
                cgene.attributes['evidence_sources'] = 'Existing'
                for etx in eg.transcripts:
                    ctx = Transcript(
                        transcript_id=etx.transcript_id,
                        seqid=etx.seqid,
                        strand=etx.strand,
                        start=etx.start,
                        end=etx.end,
                        source='Refined'
                    )
                    ctx.exons = list(etx.exons)
                    ctx.cds = list(etx.cds)
                    ctx.five_prime_utrs = list(etx.five_prime_utrs)
                    ctx.three_prime_utrs = list(etx.three_prime_utrs)
                    cgene.transcripts.append(ctx)
                consensus_genes.append(cgene)
            logger.info(f"  Loaded {len(consensus_genes)} existing gene models")

            self.tracer.snapshot("After Step 1 (refine-existing load)", consensus_genes)

            # Incorporate manual annotations: add/replace overlapping genes
            if self.manual_genes:
                consensus_genes = self._incorporate_manual_genes(consensus_genes)
                self.tracer.snapshot("After manual incorporation", consensus_genes)

            if not self.evidence_refinement:
                logger.info("\nSkipping evidence-based refinement "
                            "(use --refine_with_evidence to enable).")
                logger.info(f"  Returning {len(consensus_genes)} gene models "
                            "from existing+manual annotation as-is.")
                return consensus_genes

        else:
            # --- Full consensus mode ---
            # Step 1: Build initial consensus gene models
            logger.info("\nStep 1: Building consensus gene models...")
            consensus_genes = self._build_consensus()
            logger.info(f"  Built {len(consensus_genes)} initial consensus models")
            self.tracer.snapshot("After Step 1 (consensus)", consensus_genes)

            # Incorporate manual annotations into consensus
            if self.manual_genes:
                consensus_genes = self._incorporate_manual_genes(consensus_genes)
                self.tracer.snapshot("After manual incorporation", consensus_genes)

            # Step 1b: Split genes where StringTie/TD model separate genes
            if self.st_genes:
                logger.info("\nStep 1b: Splitting genes based on StringTie evidence...")
                split_genes = []
                for gene in consensus_genes:
                    # Don't split manually annotated genes
                    if gene.attributes.get('manual_annotation') == 'true':
                        split_genes.append(gene)
                    else:
                        sub_genes = self._split_by_stringtie(gene)
                        split_genes.extend(sub_genes)
                if len(split_genes) > len(consensus_genes):
                    logger.info(f"  Split {len(consensus_genes)} genes into {len(split_genes)}")
                consensus_genes = split_genes
                self.tracer.snapshot("After Step 1b (StringTie split)", consensus_genes)

        # Validate coordinates before processing
        logger.info("\nValidating gene coordinates...")
        n_before = len(consensus_genes)
        consensus_genes = self._validate_gene_coordinates(consensus_genes)
        n_removed = n_before - len(consensus_genes)
        if n_removed:
            logger.info(f"  Removed {n_removed} genes with malformed coordinates")
        logger.info(f"  {len(consensus_genes)} genes pass validation")

        # Step 2: Filter impossible introns and enforce canonical splice sites
        logger.info("\nStep 2: Filtering impossible introns and enforcing canonical splice sites...")
        for gene in consensus_genes:
            # Skip manual annotations — preserve their structure
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            for tx in gene.transcripts:
                # Filter tiny exons and impossible introns
                tx.exons = [e for e in tx.exons if e.length >= MIN_EXON_SIZE]
                tx.exons = filter_impossible_introns(tx.exons)
                # Merge neighboring exons with continuous RNA-seq coverage
                if self.coverage.available:
                    tx.exons = self._merge_exons_by_coverage(
                        gene.seqid, tx.exons, gene.strand)
                    # Remove zero-coverage internal exons
                    tx = remove_zero_coverage_internal_exons(
                        tx, self.coverage, self.genome, gene.seqid, gene.strand)
                # Enforce canonical splice sites (adjust ±5bp)
                tx.exons = enforce_canonical_splice_sites(
                    self.genome, gene.seqid, tx.exons, gene.strand,
                    bam_evidence=self.bam_evidence)
                # Remove exons that still create non-canonical splice sites
                tx.exons = self._remove_noncanonical_exons(
                    gene.seqid, tx.exons, gene.strand)
                # Filter CDS for minimum size
                tx.cds = [c for c in tx.cds if c.length >= MIN_EXON_SIZE]
            # Re-deduplicate after modifications
            gene.transcripts = deduplicate_isoforms(gene.transcripts)
            # Remove transcripts with non-canonical splice sites
            gene.transcripts = [
                tx for tx in gene.transcripts
                if self._all_splices_canonical(gene.seqid, tx.exons, gene.strand)
            ]
        self.tracer.snapshot("After Step 2 (canonical splice enforce)", consensus_genes)

        # Step 3: Evaluate splice sites and UTR boundaries
        logger.info("\nStep 3: Evaluating splice sites and UTR boundaries...")
        for i, gene in enumerate(consensus_genes):
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            try:
                consensus_genes[i] = self.splice_eval.evaluate_5prime_splice(gene)
            except Exception as e:
                exon_info = ""
                if gene.transcripts:
                    exons = gene.transcripts[0].sorted_exons()
                    exon_info = f" exons={[(ex.start,ex.end) for ex in exons[:4]]}"
                logger.warning(f"  Splice evaluation failed for {gene.gene_id} "
                             f"({gene.seqid}:{gene.start}-{gene.end} "
                             f"{gene.strand}{exon_info}): {e}")

        # Re-validate after splice site modifications
        consensus_genes = self._validate_gene_coordinates(consensus_genes)
        self.tracer.snapshot("After Step 3 (splice/UTR eval)", consensus_genes)

        # Step 4: Recovering UTRs from StringTie
        logger.info("\nStep 4: Recovering UTRs from StringTie...")
        if self.st_genes:
            for i, gene in enumerate(consensus_genes):
                if gene.attributes.get('manual_annotation') == 'true':
                    continue
                consensus_genes[i] = self.utr_recovery.recover_utrs(gene, self.st_genes)

        # Re-validate after UTR recovery
        consensus_genes = self._validate_gene_coordinates(consensus_genes)
        self.tracer.snapshot("After Step 4 (UTR recovery)", consensus_genes)

        # Step 5: Evaluate gene merging
        logger.info("\nStep 5: Evaluating gene merges...")
        consensus_genes = self._evaluate_merges(consensus_genes)
        logger.info(f"  After merging: {len(consensus_genes)} genes")
        self.tracer.snapshot("After Step 5 (merges)", consensus_genes)

        # Step 5b: Post-merge splice validation and cleanup.
        # Each substep is instrumented for traced genes so you can see
        # which specific cleanup dropped an exon.
        logger.info("\nStep 5b: Post-merge splice validation...")
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            for tx in gene.transcripts:
                if self.coverage.available:
                    tx = trim_zero_coverage_terminal_exons(
                        tx, self.coverage, gene.seqid, gene.strand)
                    if self.tracer.matches(gene):
                        self.tracer.snapshot(
                            "  Step 5b.1 (trim zero-cov terminal)", [gene])
                    tx = remove_zero_coverage_internal_exons(
                        tx, self.coverage, self.genome, gene.seqid, gene.strand)
                    if self.tracer.matches(gene):
                        self.tracer.snapshot(
                            "  Step 5b.2 (remove zero-cov internal)", [gene])
                tx.exons = filter_impossible_introns(tx.exons)
                if self.tracer.matches(gene):
                    self.tracer.snapshot(
                        "  Step 5b.3 (filter impossible introns)", [gene])
                tx.exons = enforce_canonical_splice_sites(
                    self.genome, gene.seqid, tx.exons, gene.strand,
                    bam_evidence=self.bam_evidence)
                if self.tracer.matches(gene):
                    self.tracer.snapshot(
                        "  Step 5b.4 (enforce canonical splice)", [gene])
                tx.exons = self._remove_noncanonical_exons(
                    gene.seqid, tx.exons, gene.strand)
                if self.tracer.matches(gene):
                    self.tracer.snapshot(
                        "  Step 5b.5 (remove noncanonical exons)", [gene])
            gene.transcripts = deduplicate_isoforms(gene.transcripts)
            gene.transcripts = [
                tx for tx in gene.transcripts
                if self._all_splices_canonical(gene.seqid, tx.exons, gene.strand)
            ]
        self.tracer.snapshot("After Step 5b (post-merge cleanup)", consensus_genes)

        # Step 5c: Merge neighboring exons with continuous RNA-seq coverage
        logger.info("\nStep 5c: Merging exons by RNA-seq coverage...")
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            if self.coverage.available:
                for tx in gene.transcripts:
                    tx.exons = self._merge_exons_by_coverage(
                        gene.seqid, tx.exons, gene.strand)
            gene.transcripts = deduplicate_isoforms(gene.transcripts)
        self.tracer.snapshot("After Step 5c (merge exons by coverage)", consensus_genes)

        # Step 5d: Remove unsupported exons (no RNA-seq coverage AND no ST/TD match)
        logger.info("\nStep 5d: Filtering unsupported exons...")
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            for tx in gene.transcripts:
                tx.exons = self._filter_unsupported_exons(
                    gene.seqid, tx.exons, gene.strand)
            if self.tracer.matches(gene):
                self.tracer.snapshot(
                    "  Step 5d.1 (filter unsupported)", [gene])
            # Remove chimeric exons: internal exons inserted into evidence introns
            for tx in gene.transcripts:
                tx.exons = self._remove_chimeric_exons(
                    gene.seqid, tx.exons, gene.strand)
            if self.tracer.matches(gene):
                self.tracer.snapshot(
                    "  Step 5d.2 (remove chimeric)", [gene])
            for tx in gene.transcripts:
                tx.exons = enforce_canonical_splice_sites(
                    self.genome, gene.seqid, tx.exons, gene.strand,
                    bam_evidence=self.bam_evidence)
            gene.transcripts = deduplicate_isoforms(gene.transcripts)
            gene.transcripts = [
                tx for tx in gene.transcripts
                if self._all_splices_canonical(gene.seqid, tx.exons, gene.strand)
            ]
        self.tracer.snapshot("After Step 5d (filter unsupported exons)", consensus_genes)

        # Step 5e: Split non-overlapping isoforms into separate genes
        logger.info("\nStep 5e: Splitting non-overlapping isoforms...")
        consensus_genes = self._split_nonoverlapping_isoforms(consensus_genes)
        self.tracer.snapshot("After Step 5e (split non-overlapping isoforms)", consensus_genes)

        # Step 5f: Rank and select isoforms
        logger.info("\nStep 5f: Ranking isoforms...")
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            if len(gene.transcripts) > 1:
                gene.transcripts = self._rank_isoforms(gene)
        self.tracer.snapshot("After Step 5f (rank isoforms)", consensus_genes)

        # Step 5f.5: Upgrade internal exon boundaries using junction evidence.
        # After phase 4 selects the winning template, individual exon boundaries
        # may still be at suboptimal positions (e.g. StringTie exon 4 bp shorter
        # than the junction-rich TransDecoder boundary). Scan ±150 bp for a
        # better-supported splice site and move the boundary outward when the
        # candidate has ≥3× more reads and ≥5 reads.
        logger.info("\nStep 5f.5: Upgrading exon boundaries using junction evidence...")
        self._upgrade_exon_boundaries(consensus_genes)
        self.tracer.snapshot("After Step 5f.5 (boundary upgrade)", consensus_genes)

        # Step 5g: Re-derive CDS from longest ORF in refined transcripts
        logger.info("\nStep 5g: Reassigning CDS from longest ORF...")
        orf_finder = ORFFinder(self.genome)
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            gene = orf_finder.reassign_cds(gene, coverage=self.coverage,
                                          evidence_index=self.evidence_index)
        self.tracer.snapshot("After Step 5g (CDS reassign)", consensus_genes)

        # Step 5g.5: Recover downstream exons for genes lacking a stop codon.
        # Now that CDS has been assigned (step 5g), we can detect stop-codon-free genes.
        # When a junction lands inside the terminal exon, that exon is trimmed
        # and the junction chain is followed until a stop codon is found.
        # CDS is cleared and re-derived in a second targeted reassign pass.
        logger.info("\nStep 5g.5: Recovering downstream exons for stop-codon-free genes...")
        self._recover_downstream_exons(consensus_genes, orf_finder)
        # Re-derive CDS only for genes that had exons added
        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            if gene.attributes.get('_downstream_recovered'):
                gene = orf_finder.reassign_cds(gene, coverage=self.coverage,
                                              evidence_index=self.evidence_index)
        self.tracer.snapshot("After Step 5g.5 (downstream recovery)", consensus_genes)

        # Step 5h: Split genes where isoforms have non-overlapping CDS
        logger.info("\nStep 5h: Splitting genes with non-overlapping CDS...")
        consensus_genes = self._split_by_cds_overlap(consensus_genes)
        self.tracer.snapshot("After Step 5h (CDS-overlap split)", consensus_genes)

        # Step 5h.4: Drop weakly-supported internal exons that introduce a
        # premature stop and bury the rest of the gene as multi-exon UTR.
        # Trigger: ≥3 UTR exons on one end of the transcript.  For each
        # internal exon, test whether removing it produces a substantially
        # longer ORF that reaches the terminal exon; if so, and if the exon
        # is poorly supported relative to the corresponding skipping
        # junction, drop it.
        logger.info("\nStep 5h.4: Dropping weak exons that shorten ORF...")
        self._drop_weak_premature_stop_exons(consensus_genes, orf_finder)
        self.tracer.snapshot("After Step 5h.4 (weak-exon drop)", consensus_genes)

        # Step 5h.5: Split merged genes flagged by excessive UTR-exon counts.
        # After CDS reassignment, a transcript with 3+ UTR exons on one end
        # is almost always two neighboring genes merged via a bridging
        # template.  Use StringTie evidence to find the real split point.
        logger.info("\nStep 5h.5: Splitting merged genes via excessive-UTR heuristic...")
        consensus_genes = self._split_excessive_utr_genes(consensus_genes)
        self.tracer.snapshot("After Step 5h.5 (UTR-triggered split)", consensus_genes)

        # Step 5i: Repair transcripts that lost exons during filtering
        logger.info("\nStep 5i: Repairing gene models and recomputing boundaries...")
        n_repaired = 0
        n_removed_tx = 0
        for gene in consensus_genes:
            for tx in gene.transcripts:
                if not tx.exons and tx.cds:
                    # Regenerate minimal exons from CDS spans
                    # Group overlapping/adjacent CDS into exon-like regions
                    sorted_cds = sorted(tx.cds, key=lambda c: c.start)
                    exon_regions = [(sorted_cds[0].start, sorted_cds[0].end)]
                    for c in sorted_cds[1:]:
                        if c.start <= exon_regions[-1][1] + 1:
                            exon_regions[-1] = (exon_regions[-1][0],
                                                max(exon_regions[-1][1], c.end))
                        else:
                            exon_regions.append((c.start, c.end))
                    tx.exons = [
                        Feature(seqid=gene.seqid, source='Refined', ftype='exon',
                                start=s, end=e, score=0.0, strand=gene.strand,
                                phase='.', attributes={})
                        for s, e in exon_regions
                    ]
                    n_repaired += 1
                    logger.debug(f"  Regenerated {len(tx.exons)} exons from CDS "
                               f"for {gene.gene_id}")

            # Remove transcripts with neither exons nor CDS
            before = len(gene.transcripts)
            gene.transcripts = [tx for tx in gene.transcripts
                               if tx.exons or tx.cds]
            n_removed_tx += before - len(gene.transcripts)

        if n_repaired:
            logger.info(f"  Regenerated exons for {n_repaired} transcripts from CDS")
        if n_removed_tx:
            logger.info(f"  Removed {n_removed_tx} empty transcripts")

        # Recompute gene and mRNA boundaries from exon coordinates
        for gene in consensus_genes:
            self._recompute_gene_boundaries(gene)

        # Final validation: remove any genes that still have invalid bounds
        before = len(consensus_genes)
        consensus_genes = [g for g in consensus_genes
                          if isinstance(g.start, int) and isinstance(g.end, int)
                          and g.start >= 1 and g.start <= g.end]
        n_invalid = before - len(consensus_genes)
        if n_invalid:
            logger.warning(f"  Removed {n_invalid} genes with invalid boundaries")
        self.tracer.snapshot("After Step 5i (repair/boundaries)", consensus_genes)

        # Step 5j: Add alternative coding isoforms from TransDecoder.
        # TD models are isoform-resolved; consensus building collapses
        # them to one transcript per gene.  Recover divergent isoforms
        # whose CDS substantially overlaps the primary CDS but whose
        # 5'/3' end or splice structure differs significantly.
        logger.info("\nStep 5j: Adding alternative coding isoforms from TransDecoder...")
        self._add_alternative_isoforms(consensus_genes, orf_finder)
        self.tracer.snapshot("After Step 5j (alt isoforms)", consensus_genes)

        # Step 5k: Merge same-strand genes that share at least one exon
        # exactly (alt-start / alt-end isoforms that came in as separate
        # consensus genes -- typically from Helixer making multiple
        # predictions over what is really one gene).
        logger.info("\nStep 5k: Merging same-strand genes sharing exact exons...")
        consensus_genes = self._merge_alt_start_genes(consensus_genes)
        self.tracer.snapshot("After Step 5k (alt-start merge)", consensus_genes)

        # Step 6: Compute posterior probabilities
        logger.info("\nStep 6: Computing posterior probabilities...")
        # Remove genes with no valid transcripts remaining
        consensus_genes = [g for g in consensus_genes if g.transcripts]

        # Diagnostic: test BAM junction support on first multi-exon gene
        if self.bam_evidence.available:
            for gene in consensus_genes:
                introns = gene.transcripts[0].introns() if gene.transcripts else []
                if introns:
                    test_intron = introns[0]
                    test_count = self.bam_evidence.count_spliced_reads(
                        gene.seqid, test_intron[0], test_intron[1])
                    logger.info(f"  BAM diagnostic: {gene.gene_id} intron "
                               f"{test_intron[0]}-{test_intron[1]} -> "
                               f"{test_count} spliced reads")
                    if test_count == 0:
                        logger.warning(f"  BAM WARNING: 0 spliced reads found. "
                                      f"Check BAM reference names match GFF seqid "
                                      f"'{gene.seqid}' and BAM is indexed.")
                    break

        for gene in consensus_genes:
            if gene.attributes.get('manual_annotation') == 'true':
                # Manual genes get high posterior automatically
                gene.posterior = 0.99
            else:
                gene.posterior = self.posterior_calc.score_gene(
                    gene, self.helixer_genes, self.td_genes, self.st_genes)
        if self.tracer.enabled:
            for g in consensus_genes:
                if self.tracer.matches(g):
                    self.tracer.event(
                        "After Step 6",
                        f"{g.gene_id} posterior={g.posterior:.3f}")

        # Step 7: Filter by posterior threshold
        logger.info("\nStep 7: Applying posterior thresholds...")
        # Manual genes always pass
        passing_genes = [g for g in consensus_genes
                        if g.posterior >= self.coding_threshold
                        or g.attributes.get('manual_annotation') == 'true']
        filtered_genes = [g for g in consensus_genes
                         if g.posterior < self.coding_threshold
                         and g.attributes.get('manual_annotation') != 'true']
        logger.info(f"  {len(passing_genes)} genes pass coding threshold ({self.coding_threshold})")
        logger.info(f"  {len(filtered_genes)} genes removed (low confidence)")
        if self.tracer.enabled:
            for g in filtered_genes:
                if self.tracer.matches(g):
                    self.tracer.event(
                        "Step 7 FILTERED",
                        f"{g.gene_id} dropped (posterior={g.posterior:.3f} "
                        f"< threshold={self.coding_threshold})")
            self.tracer.snapshot("After Step 7 (passing)", passing_genes)

        # Step 8: Detect ncRNAs
        logger.info("\nStep 8: Detecting ncRNA candidates...")
        coding_regions = [(g.start, g.end, g.strand) for g in passing_genes]
        ncrna_genes = []
        if self.st_genes and self.coverage.available:
            seqids = set(g.seqid for g in self.st_genes)
            for seqid in seqids:
                ncrna_genes.extend(self.ncrna_detector.find_ncrna_candidates(
                    seqid, self.st_genes, self.helixer_genes, self.td_genes, coding_regions
                ))
        logger.info(f"  Found {len(ncrna_genes)} ncRNA candidates")

        # Compute posteriors for ncRNAs
        for ncrna in ncrna_genes:
            ncrna.posterior = self._score_ncrna(ncrna)

        ncrna_passing = [g for g in ncrna_genes if g.posterior >= self.ncrna_threshold]
        logger.info(f"  {len(ncrna_passing)} ncRNAs pass threshold ({self.ncrna_threshold})")

        # Step 9: Compute per-feature posteriors
        logger.info("\nStep 9: Computing per-feature posteriors...")
        all_genes = passing_genes + ncrna_passing

        # Deduplicate features within each gene (may result from merging/UTR recovery)
        for gene in all_genes:
            for tx in gene.transcripts:
                tx.exons = GeneMerger._deduplicate_features(tx.exons)
                tx.cds = GeneMerger._deduplicate_features(tx.cds)
                tx.five_prime_utrs = GeneMerger._deduplicate_features(tx.five_prime_utrs)
                tx.three_prime_utrs = GeneMerger._deduplicate_features(tx.three_prime_utrs)
                # Filter out tiny features (< 3bp) that are likely artifacts
                tx.exons = [e for e in tx.exons if e.length >= 3]
                tx.cds = [c for c in tx.cds if c.length >= 3]

        for gene in all_genes:
            self._annotate_features(gene)

        # Refine gene boundaries to match actual transcript extent
        for gene in all_genes:
            if gene.transcripts:
                all_starts = []
                all_ends = []
                for tx in gene.transcripts:
                    for e in tx.exons:
                        all_starts.append(e.start)
                        all_ends.append(e.end)
                if all_starts:
                    gene.start = min(all_starts)
                    gene.end = max(all_ends)
                    # Also update transcript boundaries
                    for tx in gene.transcripts:
                        if tx.exons:
                            tx.start = min(e.start for e in tx.exons)
                            tx.end = max(e.end for e in tx.exons)

        # Sort by position
        all_genes.sort(key=lambda g: (g.seqid, g.start))

        # Assign temporary sequential gene IDs (final naming done by main
        # based on --renumber, --gene_prefix, --name_from options)
        for i, gene in enumerate(all_genes, 1):
            # Preserve manual annotation gene IDs
            if gene.attributes.get('manual_annotation') != 'true':
                gene.gene_id = f"refined_{i:06d}"
                gene.attributes['ID'] = gene.gene_id
            for j, tx in enumerate(gene.transcripts, 1):
                tx.transcript_id = f"{gene.gene_id}.{j}"

        logger.info(f"\nFinal output: {len(all_genes)} refined gene models")
        coding_count = sum(1 for g in all_genes if g.gene_type == 'protein_coding')
        ncrna_count = sum(1 for g in all_genes if g.gene_type == 'ncRNA')
        logger.info(f"  Protein-coding: {coding_count}")
        logger.info(f"  ncRNA: {ncrna_count}")

        self.tracer.snapshot("Final output (pre-renumber)", all_genes)
        return all_genes

    def _all_splices_canonical(self, seqid: str, exons: List[Feature],
                                strand: str) -> bool:
        """Check that all introns in an exon set have canonical splice sites."""
        sorted_exons = sorted(exons, key=lambda e: e.start)
        for i in range(len(sorted_exons) - 1):
            intron_s = sorted_exons[i].end + 1
            intron_e = sorted_exons[i + 1].start - 1
            if intron_e - intron_s + 1 < MIN_INTRON_SIZE:
                continue
            if strand == '+':
                donor = self.genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                acceptor = self.genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
            else:
                donor = reverse_complement(
                    self.genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                acceptor = reverse_complement(
                    self.genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()
            # At scaffold boundaries, sequences may be truncated — skip check
            if len(donor) < 2 or len(acceptor) < 2:
                continue
            if not ((donor == 'GT' and acceptor == 'AG') or
                    (donor == 'GC' and acceptor == 'AG')):
                return False
        return True

    def _filter_unsupported_exons(self, seqid: str, exons: List[Feature],
                                    strand: str) -> List[Feature]:
        """Remove internal exons that have NEITHER ST/TD support NOR RNA-seq coverage.

        An exon is kept if ANY of these are true:
        - It is the first or last exon (terminal)
        - It overlaps a StringTie or TransDecoder exon/CDS within ±10bp
        - It has RNA-seq coverage >= 5.0
        - Removing it would NOT create a canonical intron
        """
        if len(exons) < 3:
            return exons

        eidx = self.evidence_index

        sorted_exons = sorted(exons, key=lambda e: e.start)
        keep = [True] * len(sorted_exons)

        for i in range(1, len(sorted_exons) - 1):  # skip terminal exons
            if eidx.has_evidence_for_exon(seqid, strand,
                                           sorted_exons[i].start,
                                           sorted_exons[i].end):
                continue

            # Check RNA-seq coverage — keep if covered
            cov = self.coverage.get_mean_coverage(seqid,
                                                   sorted_exons[i].start,
                                                   sorted_exons[i].end)
            if cov >= 5.0:
                continue

            # No evidence AND no coverage — check if removal creates canonical intron
            skip_intron_s = sorted_exons[i - 1].end + 1
            skip_intron_e = sorted_exons[i + 1].start - 1

            if skip_intron_e - skip_intron_s + 1 < MIN_INTRON_SIZE:
                continue

            if strand == '+':
                donor = self.genome.get_sequence(seqid, skip_intron_s, skip_intron_s + 1).upper()
                acceptor = self.genome.get_sequence(seqid, skip_intron_e - 1, skip_intron_e).upper()
            else:
                donor = reverse_complement(
                    self.genome.get_sequence(seqid, skip_intron_e - 1, skip_intron_e)).upper()
                acceptor = reverse_complement(
                    self.genome.get_sequence(seqid, skip_intron_s, skip_intron_s + 1)).upper()

            # Skip check if sequence was truncated at scaffold boundary
            if len(donor) < 2 or len(acceptor) < 2:
                pass  # assume canonical at scaffold boundary
            if (donor == 'GT' and acceptor == 'AG') or (donor == 'GC' and acceptor == 'AG'):
                keep[i] = False
                logger.info(f"Removing unsupported exon {sorted_exons[i].start}-"
                           f"{sorted_exons[i].end} (no ST/TD, cov={cov:.1f})")

        return [e for e, k in zip(sorted_exons, keep) if k]

    def _remove_chimeric_exons(self, seqid: str, exons: List[Feature],
                                strand: str) -> List[Feature]:
        """Remove internal exons that are chimeric insertions into evidence introns.

        An exon is chimeric if:
        1. No evidence model (ST/TD) has it paired with either neighbor exon
        2. An evidence model DOES pair the two surrounding exons directly
           (meaning an evidence model explicitly treats this region as an intron)

        This prevents the consensus builder from inserting Helixer exons into
        introns that StringTie/TransDecoder models intentionally skip.
        """
        if len(exons) < 3:
            return exons

        eidx = self.evidence_index
        TOL = 10

        sorted_exons = sorted(exons, key=lambda e: e.start)
        keep = [True] * len(sorted_exons)

        for i in range(1, len(sorted_exons) - 1):
            prev = sorted_exons[i - 1]
            this = sorted_exons[i]
            nxt = sorted_exons[i + 1]

            # Does any evidence model pair this exon with either neighbor?
            left_paired = eidx.has_exon_pair(seqid, strand,
                                              prev.start, prev.end,
                                              this.start, this.end, TOL)
            right_paired = eidx.has_exon_pair(seqid, strand,
                                               this.start, this.end,
                                               nxt.start, nxt.end, TOL)

            if left_paired or right_paired:
                continue  # Evidence supports this exon in this context

            # No evidence pairs it with either neighbor — check if evidence
            # explicitly skips it (pairs the surrounding exons directly)
            skip_paired = eidx.has_exon_pair(seqid, strand,
                                              prev.start, prev.end,
                                              nxt.start, nxt.end, TOL)

            if skip_paired:
                # Verify removal creates a canonical intron
                intron_s = prev.end + 1
                intron_e = nxt.start - 1
                if intron_e - intron_s + 1 >= MIN_INTRON_SIZE:
                    if strand == '+':
                        d = self.genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                        a = self.genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
                    else:
                        d = reverse_complement(
                            self.genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                        a = reverse_complement(
                            self.genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()

                    if (d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG'):
                        keep[i] = False
                        logger.info(f"Removing chimeric exon {this.start}-{this.end} "
                                   f"({this.end - this.start + 1}bp): "
                                   f"inserted into evidence intron "
                                   f"{prev.end+1}-{nxt.start-1}")

        return [e for e, k in zip(sorted_exons, keep) if k]

    def _merge_exons_by_coverage(self, seqid: str, exons: List[Feature],
                                  strand: str) -> List[Feature]:
        """Merge neighboring exons when the inter-exon gap has continuous RNA-seq coverage.

        Helixer can predict exon boundaries inaccurately since it doesn't use
        RNA-seq data. When two adjacent exons have a short gap (<500bp) with
        continuous high coverage (mean > 50x, min > 10x), merge them into one exon.
        """
        if len(exons) < 2:
            return exons

        sorted_exons = sorted(exons, key=lambda e: e.start)
        merged = [sorted_exons[0]]

        for i in range(1, len(sorted_exons)):
            prev = merged[-1]
            curr = sorted_exons[i]
            gap_s = prev.end + 1
            gap_e = curr.start - 1
            gap_len = gap_e - gap_s + 1

            should_merge = False
            if 1 <= gap_len <= 500:
                # Never merge if there is junction evidence anywhere in the
                # gap — this includes both a single spanning junction AND any
                # pair of junctions that bracket a dropped middle exon.
                # (1) A junction whose donor or acceptor falls inside the gap.
                # (2) A junction that spans the whole gap (count_spliced_reads).
                junc_in_gap = (
                    self.bam_evidence.find_junctions_starting_in(
                        seqid, gap_s, gap_e, min_reads=2) or
                    self.bam_evidence.find_junctions_ending_in(
                        seqid, gap_s, gap_e, min_reads=2) or
                    self.bam_evidence.count_spliced_reads(
                        seqid, gap_s, gap_e, tolerance=3) >= 2
                )
                if junc_in_gap:
                    pass  # junction evidence in gap: do not merge
                else:
                    try:
                        gap_cov = self.coverage.get_coverage(seqid, gap_s, gap_e)
                        if len(gap_cov) > 0:
                            gap_mean = float(np.mean(gap_cov))
                            gap_min = float(np.min(gap_cov))

                            if gap_mean > 50 and gap_min > 10:
                                prev_cov = self.coverage.get_mean_coverage(seqid, prev.start, prev.end)
                                curr_cov = self.coverage.get_mean_coverage(seqid, curr.start, curr.end)
                                flank_mean = (prev_cov + curr_cov) / 2
                                if gap_mean >= flank_mean * 0.3:
                                    should_merge = True
                    except Exception:
                        pass

            if should_merge:
                # Verify neighboring introns remain canonical after merge
                ok = True
                if len(merged) >= 2:
                    intron_s = merged[-2].end + 1
                    intron_e = prev.start - 1
                    if intron_e - intron_s + 1 >= MIN_INTRON_SIZE:
                        if strand == '+':
                            d = self.genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                            a = self.genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
                        else:
                            d = reverse_complement(self.genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                            a = reverse_complement(self.genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()
                        if not ((d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG')):
                            ok = False

                if i + 1 < len(sorted_exons) and ok:
                    intron_s = curr.end + 1
                    intron_e = sorted_exons[i + 1].start - 1
                    if intron_e - intron_s + 1 >= MIN_INTRON_SIZE:
                        if strand == '+':
                            d = self.genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                            a = self.genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
                        else:
                            d = reverse_complement(self.genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                            a = reverse_complement(self.genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()
                        if not ((d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG')):
                            ok = False

                if ok:
                    logger.info(f"Merging exons {prev.start}-{prev.end} + {curr.start}-{curr.end} "
                               f"(gap {gap_len}bp, cov={gap_mean:.0f})")
                    merged[-1] = Feature(
                        seqid=prev.seqid, source=prev.source, ftype=prev.ftype,
                        start=prev.start, end=curr.end,
                        score=prev.score, strand=prev.strand, phase=prev.phase,
                        attributes=prev.attributes
                    )
                    continue

            merged.append(curr)

        return merged

    def _rank_isoforms(self, gene: 'Gene') -> List['Transcript']:
        """Remove clearly inferior isoforms while preserving genuine alternatives.

        Keep isoforms that:
        - Have different first CDS (alternative start sites / promoters)
        - Have different exon count (exon skipping)
        - Have structurally different internal exons
        Remove only isoforms where:
        - Same exon set but minor boundary shifts, with worse coverage at diffs
        """
        if len(gene.transcripts) <= 1:
            return gene.transcripts

        scored = []
        for tx in gene.transcripts:
            total_cov = 0.0
            total_bp = 0
            for exon in tx.exons:
                bp = exon.end - exon.start + 1
                cov = self.coverage.get_mean_coverage(gene.seqid, exon.start, exon.end)
                total_cov += cov * bp
                total_bp += bp
            mean_cov = total_cov / total_bp if total_bp > 0 else 0
            evidence_count = self._count_supported_exons(tx.exons)
            n_exons = len(tx.exons)
            scored.append((tx, mean_cov, evidence_count, n_exons))

        scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
        best_tx, best_cov, best_ev, best_n = scored[0]
        kept = [best_tx]

        best_exon_set = set((e.start, e.end) for e in best_tx.exons)

        for tx, cov, ev, n_exons in scored[1:]:
            this_exon_set = set((e.start, e.end) for e in tx.exons)

            # Different number of exons -> likely genuine isoform
            if n_exons != best_n:
                kept.append(tx)
                continue

            # Same number of exons — check if just boundary shifts
            shared = best_exon_set & this_exon_set
            only_this = this_exon_set - shared
            only_best = best_exon_set - shared

            if len(only_this) == 0:
                # Identical -> skip (already deduplicated, but safety)
                continue

            if len(only_this) >= 3:
                # Many differing exons -> genuinely different
                kept.append(tx)
                continue

            # 1-2 differing exons: check if they're just boundary shifts
            # by seeing if each "only_this" exon is close to an "only_best" exon
            boundary_shifts = 0
            for ts, te in only_this:
                for bs, be in only_best:
                    if abs(ts - bs) <= 150 and abs(te - be) <= 150:
                        boundary_shifts += 1
                        break

            if boundary_shifts == len(only_this):
                # All differences are boundary shifts -> drop the worse one
                diff_cov_this = sum(
                    self.coverage.get_mean_coverage(gene.seqid, s, e) * (e - s + 1)
                    for s, e in only_this)
                diff_bp_this = sum(e - s + 1 for s, e in only_this)
                diff_cov_best = sum(
                    self.coverage.get_mean_coverage(gene.seqid, s, e) * (e - s + 1)
                    for s, e in only_best)
                diff_bp_best = sum(e - s + 1 for s, e in only_best)

                # Drop if best has more supported bases OR better total coverage
                if diff_bp_best >= diff_bp_this or diff_cov_best >= diff_cov_this:
                    logger.info(f"  Dropping boundary-shift isoform "
                               f"({diff_bp_this}bp vs {diff_bp_best}bp at diff exons)")
                    continue

            # Otherwise keep
            kept.append(tx)

        if len(kept) < len(gene.transcripts):
            logger.info(f"Gene {gene.gene_id}: reduced from {len(gene.transcripts)} "
                       f"to {len(kept)} isoforms")
        return kept

    def _count_supported_exons(self, exons: List[Feature]) -> int:
        """Count how many exons have StringTie or TransDecoder evidence."""
        if not exons:
            return 0
        seqid = exons[0].seqid
        strand = exons[0].strand
        return self.evidence_index.count_supported_exons(seqid, strand, exons)

    def _split_nonoverlapping_isoforms(self, genes: List[Gene]) -> List[Gene]:
        """Split genes whose isoforms don't share any exonic overlap into separate genes.

        When a Helixer gene covers 2+ separate ST/TD genes, the consensus builder
        creates isoforms for each. If those isoforms cover entirely different genomic
        regions (no shared exons, no overlap), they should be separate genes.
        """
        new_genes = []
        gene_counter = 0

        for gene in genes:
            if len(gene.transcripts) <= 1:
                new_genes.append(gene)
                continue

            # Group isoforms by overlap
            # Two isoforms overlap if any of their exons overlap
            # Skip transcripts with no exons
            groups = []
            for tx in gene.transcripts:
                if not tx.exons:
                    continue
                tx_interval = (min(e.start for e in tx.exons),
                               max(e.end for e in tx.exons))
                placed = False
                for group in groups:
                    for existing_tx in group:
                        ex_interval = (min(e.start for e in existing_tx.exons),
                                       max(e.end for e in existing_tx.exons))
                        # Check overlap
                        if (tx_interval[0] <= ex_interval[1] and
                                tx_interval[1] >= ex_interval[0]):
                            group.append(tx)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    groups.append([tx])

            if len(groups) <= 1:
                new_genes.append(gene)
            else:
                logger.info(f"Splitting {gene.gene_id} into {len(groups)} genes "
                           f"(non-overlapping isoforms)")
                for group in groups:
                    new_gene = Gene(
                        gene_id=gene.gene_id,  # Will be renumbered later
                        seqid=gene.seqid,
                        strand=gene.strand,
                        start=min(e.start for tx in group for e in tx.exons),
                        end=max(e.end for tx in group for e in tx.exons),
                        source=gene.source,
                        transcripts=group,
                        gene_type=gene.gene_type,
                        posterior=gene.posterior,
                        attributes=dict(gene.attributes)
                    )
                    new_genes.append(new_gene)

        return new_genes

    def _split_by_cds_overlap(self, genes: List[Gene]) -> List[Gene]:
        """Split genes where isoforms have non-overlapping CDS regions.

        When two isoforms share exons but encode proteins in completely different
        genomic regions, they represent distinct genes that Helixer bridged together.
        Split them into separate genes, trimming each transcript's exons to only
        those encoding or flanking its CDS (first CDS exon through last CDS exon).
        """
        new_genes = []

        for gene in genes:
            if len(gene.transcripts) <= 1:
                new_genes.append(gene)
                continue

            # Get CDS span for each isoform
            cds_spans = {}
            for tx in gene.transcripts:
                if tx.cds:
                    cds_spans[tx.transcript_id] = (
                        min(c.start for c in tx.cds),
                        max(c.end for c in tx.cds))

            if len(cds_spans) < 2:
                new_genes.append(gene)
                continue

            # Group isoforms by CDS overlap
            tx_by_id = {tx.transcript_id: tx for tx in gene.transcripts}
            groups = []
            for tx in gene.transcripts:
                if tx.transcript_id not in cds_spans:
                    continue
                span = cds_spans[tx.transcript_id]
                placed = False
                for group in groups:
                    for existing_id in group:
                        if existing_id not in cds_spans:
                            continue
                        ex_span = cds_spans[existing_id]
                        if span[0] <= ex_span[1] and span[1] >= ex_span[0]:
                            group.append(tx.transcript_id)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    groups.append([tx.transcript_id])

            if len(groups) <= 1:
                new_genes.append(gene)
                continue

            # Non-overlapping CDS groups — split and trim exons
            logger.info(f"Splitting {gene.gene_id} into {len(groups)} genes "
                       f"(non-overlapping CDS)")

            for group_ids in groups:
                group_txs = []
                for tid in group_ids:
                    tx = tx_by_id[tid]
                    if not tx.cds:
                        continue

                    cds_start = min(c.start for c in tx.cds)
                    cds_end = max(c.end for c in tx.cds)

                    # Keep only exons that overlap or are between CDS exons
                    sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                    first_cds_idx = None
                    last_cds_idx = None
                    for i, e in enumerate(sorted_exons):
                        if e.end >= cds_start and e.start <= cds_end:
                            if first_cds_idx is None:
                                first_cds_idx = i
                            last_cds_idx = i

                    if first_cds_idx is None:
                        continue

                    trimmed_exons = sorted_exons[first_cds_idx:last_cds_idx + 1]

                    # Build new transcript with trimmed exons
                    new_tx = Transcript(
                        transcript_id=tx.transcript_id,
                        seqid=tx.seqid,
                        strand=tx.strand,
                        start=min(e.start for e in trimmed_exons),
                        end=max(e.end for e in trimmed_exons),
                        source=tx.source
                    )
                    new_tx.exons = trimmed_exons
                    new_tx.cds = tx.cds  # CDS stays the same
                    group_txs.append(new_tx)

                if group_txs:
                    new_gene = Gene(
                        gene_id=gene.gene_id,
                        seqid=gene.seqid,
                        strand=gene.strand,
                        start=min(e.start for tx in group_txs for e in tx.exons),
                        end=max(e.end for tx in group_txs for e in tx.exons),
                        source=gene.source,
                        transcripts=group_txs,
                        gene_type=gene.gene_type,
                        posterior=gene.posterior,
                        attributes=dict(gene.attributes)
                    )
                    new_genes.append(new_gene)

        return new_genes

    def _upgrade_exon_boundaries(self, genes: List[Gene]) -> None:
        """Post-assembly pass: correct internal exon boundaries using junction evidence.

        Operates on each intron as a unit: for the intron between exon_a and exon_b,
        scan ±MAX_UPGRADE_BP around the current donor position for a junction with
        significantly more reads than the current site.  When a better junction is
        found, both exon_a.end and exon_b.start are updated to the new intron
        boundaries simultaneously, preventing overlaps.

        This fixes cases where the winning template carries an internal boundary
        that is poorly supported relative to a nearby junction-rich site.
        """
        if not self.bam_evidence.available:
            return

        MAX_UPGRADE_BP = 150      # maximum distance to search for a better site
        MIN_JUNCTION_READS = 5    # candidate must have at least this many reads
        MIN_READS_RATIO = 3.0     # candidate must have ≥ 3× the current site reads
        COORD_TOL = 3             # tolerance for portcullis ±1 coordinate ambiguity
        n_upgraded = 0

        for gene in genes:
            seqid = gene.seqid
            for tx in gene.transcripts:
                if len(tx.exons) < 2:
                    continue
                sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                new_exons = list(sorted_exons)

                # Iterate over each intron (adjacent exon pair in genomic order)
                for i in range(len(sorted_exons) - 1):
                    exon_a = new_exons[i]      # lower genomic coords
                    exon_b = new_exons[i + 1]  # higher genomic coords

                    # Current intron: exon_a.end+1 to exon_b.start-1
                    cur_intron_start = exon_a.end + 1
                    cur_reads = self.bam_evidence.count_spliced_reads(
                        seqid, cur_intron_start, exon_b.start - 1,
                        tolerance=COORD_TOL)

                    # Find the best nearby junction (by read count)
                    best_reads = cur_reads
                    best_js, best_je = cur_intron_start, exon_b.start - 1

                    candidates = self.bam_evidence.find_junctions_starting_in(
                        seqid,
                        max(1, exon_a.start + 1),   # don't search before exon_a interior
                        exon_a.end + MAX_UPGRADE_BP,
                        min_reads=MIN_JUNCTION_READS)

                    for j_start, j_end, j_count in candidates:
                        # The new intron must start within exon_a and end within exon_b
                        if not (exon_a.start < j_start <= exon_a.end + MAX_UPGRADE_BP):
                            continue
                        if not (exon_b.start - MAX_UPGRADE_BP <= j_end < exon_b.end):
                            continue
                        if j_count > best_reads:
                            best_reads = j_count
                            best_js, best_je = j_start, j_end

                    # Upgrade if the best junction is significantly better
                    if (best_reads >= MIN_JUNCTION_READS and
                            (cur_reads == 0 or best_reads >= cur_reads * MIN_READS_RATIO) and
                            (best_js != cur_intron_start or best_je != exon_b.start - 1)):
                        new_a_end = best_js - 1
                        new_b_start = best_je + 1
                        # Safety: don't invert or produce a degenerate exon
                        if new_a_end < exon_a.start or new_b_start > exon_b.end:
                            continue
                        new_exons[i] = Feature(
                            seqid=exon_a.seqid, source=exon_a.source,
                            ftype=exon_a.ftype,
                            start=exon_a.start, end=new_a_end,
                            score=exon_a.score, strand=exon_a.strand,
                            phase=exon_a.phase, attributes=exon_a.attributes)
                        new_exons[i + 1] = Feature(
                            seqid=exon_b.seqid, source=exon_b.source,
                            ftype=exon_b.ftype,
                            start=new_b_start, end=exon_b.end,
                            score=exon_b.score, strand=exon_b.strand,
                            phase=exon_b.phase, attributes=exon_b.attributes)
                        n_upgraded += 1
                        logger.info(
                            f"  Upgraded intron {gene.gene_id} [{i}→{i+1}]: "
                            f"{cur_intron_start}-{exon_b.start-1} "
                            f"→ {best_js}-{best_je} "
                            f"({cur_reads}→{best_reads} reads)")

                tx.exons = new_exons

        if n_upgraded:
            logger.info(f"  Upgraded {n_upgraded} intron boundary(ies)")

    def _recover_downstream_exons(self, genes: List[Gene],
                                   orf_finder: 'ORFFinder') -> None:
        """Recover exons downstream of the current terminal exon.

        Triggered when the terminal exon has a hidden intron, detected by
        either of two criteria (checked independently):
          1. Junction evidence whose acceptor (j_end for − strand) lands
             *inside* the terminal exon from *outside* — the terminal exon
             is too long and the chain continues downstream.
          2. The CDS has no stop codon — a catch-all for genes where the
             above junction signal is below the read threshold.

        Coverage gate — the key improvement:
          Instead of comparing each new exon only against the terminal
          (possibly atypical) anchor exon, the gate is computed from the
          full set of existing internal exons.  Specifically we use the
          10th percentile of mean coverages across all non-terminal exons
          as a floor.  This makes the gate robust to genes with variable
          per-exon expression and prevents false negatives when the
          terminal exon has unusually low coverage.

        Algorithm:
          a. Scan terminal exon for hidden introns using junction evidence.
             For − strand: junctions with j_end inside and j_start outside
             the terminal exon.  For + strand: junctions with j_start inside
             and j_end outside.  Score candidates by read count × PWM quality
             of the splice donor site; take the highest-scoring hidden intron.
          b. Trim the terminal exon at the hidden intron boundary.
          c. Follow the junction chain downstream: from the trimmed exon's
             donor position find the best junction, land at the next acceptor,
             estimate the new exon's far end from the next departing junction
             (or coverage fall-off if no further junction found).
          d. Coverage-gate each candidate new exon against the 10th-percentile
             floor derived from existing exons.
          e. Continue until a stop codon is found or no further junctions
             exceed MIN_DOWNSTREAM_READS.
        """
        if not self.bam_evidence.available:
            return

        MIN_DOWNSTREAM_READS = 3   # minimum junction reads to follow
        MAX_ITERATIONS       = 20  # safety: never add more than this many new exons

        def _exon_coverage_floor(exons_list: list, seqid: str) -> float:
            """10th-percentile mean coverage across the supplied exons (floor 1.0)."""
            covs = sorted(
                self.coverage.get_mean_coverage(seqid, e.start, e.end)
                for e in exons_list
            )
            if not covs:
                return 1.0
            p10_idx = max(0, int(len(covs) * 0.10) - 1)
            return max(1.0, covs[p10_idx])

        def _max_search_window(exons_list: list) -> int:
            """Compute a downstream search cap from intron lengths in the gene.

            Returns the median intron length (floor 2 kb) to prevent the
            exon-start lookup from jumping to a junction that belongs to a
            completely different gene far upstream/downstream.
            Falls back to 20 kb if the gene has no introns.
            """
            if len(exons_list) < 2:
                return 20_000
            se = sorted(exons_list, key=lambda e: e.start)
            intron_lengths = [se[i+1].start - se[i].end - 1
                              for i in range(len(se)-1)
                              if se[i+1].start - se[i].end - 1 > 0]
            if not intron_lengths:
                return 20_000
            intron_lengths.sort()
            median = intron_lengths[len(intron_lengths) // 2]
            return max(2_000, median)

        n_recovered = 0
        for gene in genes:
            seqid  = gene.seqid
            strand = gene.strand
            for tx in gene.transcripts:
                if not tx.exons:
                    continue

                sorted_exons = sorted(tx.exons, key=lambda e: e.start)


                # Identify terminal exon (3' end in transcript direction)
                terminal_exon = sorted_exons[0] if strand == '-' else sorted_exons[-1]
                internal_exons = sorted_exons[1:] if strand == '-' else sorted_exons[:-1]

                # ── Primary trigger: junction evidence of hidden intron ──────
                # For − strand: find junctions whose intron END (j_end) lands
                # inside the terminal exon and whose intron START (j_start) is
                # outside (below) — the "extra" sequence to the left is intron.
                # For + strand: find junctions whose intron START is inside and
                # intron END is outside (above).
                if strand == '-':
                    hidden_juncs = self.bam_evidence.find_junctions_ending_in(
                        seqid,
                        terminal_exon.start + 1,
                        terminal_exon.end - 1,
                        min_reads=MIN_DOWNSTREAM_READS)
                    hidden_juncs = [(js, je, jc) for js, je, jc in hidden_juncs
                                    if js < terminal_exon.start]
                else:
                    hidden_juncs_raw = self.bam_evidence.find_junctions_starting_in(
                        seqid,
                        terminal_exon.start + 1,
                        terminal_exon.end - 1,
                        min_reads=MIN_DOWNSTREAM_READS)
                    hidden_juncs = [(js, je, jc) for js, je, jc in hidden_juncs_raw
                                    if je > terminal_exon.end]

                # ── Secondary trigger: CDS lacks stop codon ─────────────────
                no_stop = tx.cds and not has_stop_codon(
                    self.genome, seqid, tx.cds, strand)

                if not hidden_juncs and not no_stop:
                    continue

                # Score hidden junction candidates by reads × donor PWM.
                # For − strand the donor site (5' end of intron in transcript
                # direction) is at genomic position j_start (low end of intron),
                # and we score the sequence just upstream of j_start on the
                # minus strand.  For + strand the donor is at j_start (high end
                # of exon) on the plus strand.
                best_hidden = None
                if hidden_juncs:
                    # Hard canonical filter: only follow hidden introns whose
                    # donor AND acceptor dinucleotides are canonical (GT/GC-AG).
                    # Portcullis retains some non-canonical junctions (e.g. read
                    # depth overrides the canonical filter) that produce
                    # spurious intron splits in the terminal exon — see 020810's
                    # AA-AG intron at 77934853-77935162 with 10 reads.
                    scored = []
                    for js, je, jc in hidden_juncs:
                        if strand == '-':
                            # For - strand: intron is to the LEFT. Donor (5'->3'
                            # transcript direction) is at the HIGH genomic end of
                            # the intron (je = last intron base, exon starts at
                            # je+1).  Acceptor is at the LOW end (js = first
                            # intron base, upstream exon ends at js-1).
                            donor_di = reverse_complement(
                                self.genome.get_sequence(seqid, je - 1, je)).upper()
                            acceptor_di = reverse_complement(
                                self.genome.get_sequence(seqid, js, js + 1)).upper()
                        else:
                            # For + strand: intron runs left→right.  Donor at
                            # (js, js+1) and acceptor at (je-1, je).
                            donor_di = self.genome.get_sequence(
                                seqid, js, js + 1).upper()
                            acceptor_di = self.genome.get_sequence(
                                seqid, je - 1, je).upper()
                        if len(donor_di) < 2 or len(acceptor_di) < 2:
                            continue  # scaffold truncation — skip
                        if donor_di not in ('GT', 'GC') or acceptor_di != 'AG':
                            continue  # non-canonical — skip

                        if strand == '-':
                            donor_exon_end = je + 1
                            ss_seq = self.genome.get_sequence(
                                seqid,
                                donor_exon_end - DONOR_INTRON_BP,
                                donor_exon_end + DONOR_EXON_BP - 1)
                            pwm = score_donor(reverse_complement(ss_seq))
                        else:
                            ss_seq = self.genome.get_sequence(
                                seqid,
                                js - DONOR_EXON_BP,
                                js + DONOR_INTRON_BP - 1)
                            pwm = score_donor(ss_seq)
                        scored.append((js, je, jc, pwm))
                    # Primary sort by reads, secondary by PWM
                    scored.sort(key=lambda x: (-x[2], -x[3]))
                    if scored:
                        best_hidden = scored[0]

                # If no junction-based hidden intron but secondary trigger
                # (no stop codon), proceed without trimming.
                do_trim = best_hidden is not None

                if strand == '-':
                    if do_trim:
                        _, trim_end, _, _ = best_hidden  # j_end = last base of intron
                        # Real terminal exon: the high-coord fragment after the intron
                        new_term = Feature(
                            seqid=terminal_exon.seqid,
                            source=terminal_exon.source,
                            ftype=terminal_exon.ftype,
                            start=trim_end + 1, end=terminal_exon.end,
                            score=terminal_exon.score,
                            strand=terminal_exon.strand,
                            phase=terminal_exon.phase,
                            attributes=terminal_exon.attributes)
                        sorted_exons[0] = new_term
                        terminal_exon = new_term
                    # current_donor: the low-coord end of the trimmed terminal exon.
                    # For − strand the intron departs from terminal_exon.start toward
                    # lower genomic coords; reads_at_donor queries intron_start = pos
                    # i.e. pos = terminal_exon.start (intron starts here going left).
                    current_donor = terminal_exon.start
                else:
                    if do_trim:
                        trim_start, _, _, _ = best_hidden
                        new_term = Feature(
                            seqid=terminal_exon.seqid,
                            source=terminal_exon.source,
                            ftype=terminal_exon.ftype,
                            start=terminal_exon.start, end=trim_start - 1,
                            score=terminal_exon.score,
                            strand=terminal_exon.strand,
                            phase=terminal_exon.phase,
                            attributes=terminal_exon.attributes)
                        sorted_exons[-1] = new_term
                        terminal_exon = new_term
                    current_donor = terminal_exon.end

                # Coverage floor: 10th percentile of all internal exon coverages.
                # Fall back to terminal exon coverage if no internal exons.
                ref_exons = internal_exons if internal_exons else [terminal_exon]
                cov_floor = _exon_coverage_floor(ref_exons, seqid)
                # Max search window for exon-boundary lookup: 2× max intron length.
                max_window = _max_search_window(sorted_exons)

                # ── Steps (b–e): follow junction chain downstream ───────────
                new_exons = []
                for _ in range(MAX_ITERATIONS):
                    if strand == '+':
                        # Find junctions departing from current_donor.
                        # On + strand: intron_start = current_donor + 1, so
                        # reads_at_donor(current_donor) checks for junctions where
                        # j_start = current_donor + 1.
                        next_reads = self.bam_evidence.reads_at_donor(
                            seqid, current_donor, tolerance=2)
                        if next_reads < MIN_DOWNSTREAM_READS:
                            break
                        # Find the specific junction departing at current_donor+1
                        # (within ±3 bp tolerance).  Use find_junctions_starting_in
                        # capped to max_window to avoid cross-gene contamination.
                        fwd_juncs = self.bam_evidence.find_junctions_starting_in(
                            seqid,
                            current_donor + 1 - 3,
                            current_donor + 1 + 3,
                            min_reads=MIN_DOWNSTREAM_READS)
                        if not fwd_juncs:
                            break
                        # Take the junction with the most reads (normally just one)
                        best_junc = max(fwd_juncs, key=lambda x: x[2])
                        _, j_end, j_count = best_junc
                        exon_start = j_end + 1
                        # Find the end of this new exon: the most adjacent departing
                        # junction (lowest j_start > exon_start) defines the boundary.
                        right_juncs = self.bam_evidence.find_junctions_starting_in(
                            seqid,
                            exon_start + 1,
                            exon_start + max_window,
                            min_reads=MIN_DOWNSTREAM_READS)
                        right_juncs = [(js, je, jc) for js, je, jc in right_juncs
                                       if js > exon_start]
                        if right_juncs:
                            next_junc = min(right_juncs, key=lambda x: x[0])
                            exon_end = next_junc[0] - 1
                        else:
                            exon_end = exon_start + 200
                        current_donor = exon_end

                    else:  # strand == '-'
                        # For − strand, "downstream in transcript" = lower genomic coord.
                        # current_donor = exon.start of the current exon (low-coord end).
                        # The intron to its LEFT has j_end = current_donor - 1.
                        # Use reads_at_acceptor(current_donor) rather than
                        # reads_at_donor(current_donor - 1): on minus strand the
                        # donor dinucleotide of this intron is at the exon's
                        # LEFT boundary (current_donor-1 = j_end = last intron base).
                        next_reads = self.bam_evidence.reads_at_acceptor(
                            seqid, current_donor, tolerance=2)
                        if next_reads < MIN_DOWNSTREAM_READS:
                            break
                        # Find the specific junction: j_end ≈ current_donor - 1
                        candidates = self.bam_evidence.find_junctions_ending_in(
                            seqid,
                            current_donor - 1 - 3,
                            current_donor - 1 + 3,
                            min_reads=MIN_DOWNSTREAM_READS)
                        if not candidates:
                            break
                        # Among candidates take the one with most reads
                        best_junc = max(candidates, key=lambda x: x[2])
                        j_start, _, j_count = best_junc
                        exon_end = j_start - 1   # exon ends just before intron start
                        # Find the start of this new exon: the intron further left
                        # that defines the exon's left boundary.
                        # Use the junction with the HIGHEST j_end (most adjacent to
                        # exon_end) rather than highest read count — this prevents a
                        # more distal high-coverage junction from mis-specifying the
                        # exon start.
                        left_juncs = self.bam_evidence.find_junctions_ending_in(
                            seqid,
                            max(1, exon_end - max_window),
                            exon_end - 1,
                            min_reads=MIN_DOWNSTREAM_READS)
                        # Filter to plausible exon sizes and pick the most
                        # proximal (highest j_end) whose proposed exon start
                        # position has coverage consistent with the gene.
                        # A junction from a different gene can land in the
                        # window but its exon start will have lower coverage
                        # (or the region between the exon start and exon_end
                        # will span two coverage plateaus separated by a gap).
                        plausible_left = []
                        for js, je, jc in left_juncs:
                            cand_start = je + 1
                            exon_len = exon_end - cand_start + 1
                            if exon_len > 5_000:
                                continue
                            # Require consistent coverage across the whole
                            # proposed exon body (mean > floor × 0.15).
                            body_cov = self.coverage.get_mean_coverage(
                                seqid, cand_start, exon_end)
                            if body_cov >= cov_floor * 0.15:
                                plausible_left.append((js, je, jc))
                        if plausible_left:
                            next_junc = max(plausible_left, key=lambda x: x[1])
                            exon_start = next_junc[1] + 1
                        else:
                            # No junction evidence: estimate start from coverage.
                            # Walk left from exon_end until coverage drops below
                            # the gene floor; cap at 500 bp to avoid runaway.
                            exon_start = exon_end
                            for pos in range(exon_end, max(1, exon_end - 500), -1):
                                pos_cov = self.coverage.get_mean_coverage(
                                    seqid, pos, pos)
                                if pos_cov < cov_floor * 0.15:
                                    break
                                exon_start = pos
                        current_donor = exon_start

                    # Coverage gate against the reference exon floor
                    exon_cov = self.coverage.get_mean_coverage(
                        seqid, exon_start, exon_end)
                    if exon_cov < cov_floor * 0.15:
                        logger.debug(
                            f"  Downstream exon {exon_start}-{exon_end} failed "
                            f"coverage gate ({exon_cov:.1f} < "
                            f"{cov_floor * 0.15:.1f}, floor={cov_floor:.1f})")
                        break

                    # Stranded veto: if same-strand coverage is near-zero,
                    # the unstranded support is antisense from a neighbor.
                    if not self._passes_strand_check(
                            seqid, exon_start, exon_end, strand, exon_cov):
                        logger.info(
                            f"  Downstream exon {exon_start}-{exon_end} for "
                            f"{gene.gene_id} rejected as antisense "
                            f"(stranded veto)")
                        break

                    new_exon = Feature(
                        seqid=seqid, source='Refined', ftype='exon',
                        start=exon_start, end=exon_end,
                        score=0.5, strand=strand,
                        phase='.', attributes={'sources': 'junction_recovery'})
                    new_exons.append(new_exon)
                    logger.info(
                        f"  Recovered downstream exon {exon_start}-{exon_end} "
                        f"for {gene.gene_id} ({j_count} reads, cov={exon_cov:.1f})")

                    # Check if we now have a stop codon
                    test_exons = sorted_exons + new_exons
                    # Quick stop codon check via find_best_orf on extended set
                    ev_starts = self.evidence_index.get_evidence_cds_starts(
                        seqid, strand, gene.start, gene.end)
                    test_orf = orf_finder.find_best_orf(
                        seqid, test_exons, strand,
                        coverage=self.coverage,
                        evidence_cds_starts=ev_starts)
                    if test_orf is not None:
                        new_cds_list = orf_finder.orf_to_genomic_cds(
                            seqid, test_exons, strand,
                            test_orf[0], test_orf[1])
                        if new_cds_list and has_stop_codon(
                                self.genome, seqid, new_cds_list, strand):
                            logger.debug(
                                f"  Stop codon found after {len(new_exons)} "
                                f"recovered exon(s) for {gene.gene_id}")
                            break

                if new_exons:
                    tx.exons = sorted_exons + new_exons
                    tx.cds = []  # will be re-derived after this step
                    tx.five_prime_utrs = []
                    tx.three_prime_utrs = []
                    gene.attributes['_downstream_recovered'] = True
                    n_recovered += 1
                    logger.info(
                        f"  {gene.gene_id}: recovered {len(new_exons)} downstream "
                        f"exon(s) (trimmed terminal exon + junction chain)")

        if n_recovered:
            logger.info(f"  Recovered downstream exons for {n_recovered} gene(s)")

    def _passes_strand_check(self, seqid: str, start: int, end: int,
                              strand: str, unstranded_mean: float) -> bool:
        """Veto unstranded support only when stranded data shows clear
        antisense dominance.

        Returns True (pass) if stranded data is unavailable, OR same-strand
        coverage is substantial, OR neither strand resolves clearly (sparse
        stranded data). Returns False (veto) only when antisense coverage
        is positively dominant: antisense >= 1.0 absolute AND antisense >=
        5 * sense. This avoids false vetoes in regions where stranded
        depth is low but the data is not actually antisense.
        """
        if not self.stranded_coverage.available:
            return True
        sense = self.stranded_coverage.sense_mean(seqid, start, end, strand)
        if sense is None:
            return True
        # Clear sense support — pass.
        if sense >= max(0.5, 0.10 * unstranded_mean):
            return True
        # Otherwise, only veto if there's positive antisense evidence.
        antisense = self.stranded_coverage.antisense_mean(
            seqid, start, end, strand)
        if antisense is None:
            return True
        if antisense >= 1.0 and antisense >= 5.0 * max(sense, 0.1):
            return False
        # Insufficient stranded data to call — pass.
        return True

    @staticmethod
    def _recompute_gene_boundaries(gene: Gene):
        """Recompute gene and mRNA start/end from actual exon coordinates."""
        if not gene.transcripts:
            return

        gene_start = float('inf')
        gene_end = 0
        for tx in gene.transcripts:
            if tx.exons:
                tx_start = min(e.start for e in tx.exons)
                tx_end = max(e.end for e in tx.exons)
                tx.start = tx_start
                tx.end = tx_end
                gene_start = min(gene_start, tx_start)
                gene_end = max(gene_end, tx_end)

        # If no transcripts had exons, fall back to CDS or existing bounds
        if gene_start == float('inf'):
            all_coords = []
            for tx in gene.transcripts:
                for c in tx.cds:
                    all_coords.append((c.start, c.end))
            if all_coords:
                gene_start = min(s for s, e in all_coords)
                gene_end = max(e for s, e in all_coords)
            else:
                # No exons, no CDS — leave existing bounds unchanged
                return

        gene.start = gene_start
        gene.end = gene_end

    def _remove_noncanonical_exons(self, seqid: str, exons: List[Feature],
                                     strand: str) -> List[Feature]:
        """Remove internal exons that create non-canonical splice sites
        even after enforce_canonical_splice_sites has tried adjusting.

        Iteratively checks all introns. If any intron has a non-canonical
        splice site, removes the shorter of its flanking exons (unless it's
        terminal) and re-checks.
        """
        if len(exons) < 2:
            return exons

        changed = True
        max_iterations = 10
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            sorted_exons = sorted(exons, key=lambda e: e.start)
            to_remove = set()

            for i in range(len(sorted_exons) - 1):
                intron_s = sorted_exons[i].end + 1
                intron_e = sorted_exons[i + 1].start - 1

                if intron_e - intron_s + 1 < MIN_INTRON_SIZE:
                    continue

                if strand == '+':
                    donor = self.genome.get_sequence(seqid, intron_s, intron_s + 1).upper()
                    acceptor = self.genome.get_sequence(seqid, intron_e - 1, intron_e).upper()
                else:
                    donor = reverse_complement(
                        self.genome.get_sequence(seqid, intron_e - 1, intron_e)).upper()
                    acceptor = reverse_complement(
                        self.genome.get_sequence(seqid, intron_s, intron_s + 1)).upper()

                # Skip check if sequence was truncated at scaffold boundary
                if len(donor) < 2 or len(acceptor) < 2:
                    pass  # assume canonical at scaffold boundary
                is_ok = ((donor == 'GT' and acceptor == 'AG') or
                         (donor == 'GC' and acceptor == 'AG'))
                if not is_ok:
                    # Decide which exon to remove:
                    # - Don't remove terminal exons
                    # - Remove the one with less coverage
                    # - Prefer removing the shorter one
                    if i == 0:
                        # Can't remove first, remove second if not last
                        if i + 1 < len(sorted_exons) - 1:
                            to_remove.add(i + 1)
                    elif i + 1 == len(sorted_exons) - 1:
                        # Can't remove last, remove the one before
                        if i > 0:
                            to_remove.add(i)
                    else:
                        # Internal: remove the shorter one
                        len_a = sorted_exons[i].length
                        len_b = sorted_exons[i + 1].length
                        cov_a = self.coverage.get_mean_coverage(seqid, sorted_exons[i].start, sorted_exons[i].end)
                        cov_b = self.coverage.get_mean_coverage(seqid, sorted_exons[i + 1].start, sorted_exons[i + 1].end)
                        if cov_a < cov_b or (cov_a == cov_b and len_a <= len_b):
                            to_remove.add(i)
                        else:
                            to_remove.add(i + 1)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    logger.debug(f"Removing exon {sorted_exons[idx].start}-{sorted_exons[idx].end} "
                               f"(non-canonical splice site)")
                exons = [e for j, e in enumerate(sorted_exons) if j not in to_remove]
                changed = True
            else:
                exons = sorted_exons

        return exons

    def _build_consensus(self) -> List[Gene]:
        """Build consensus gene models from evidence-weighted exon assembly.

        Strategy — evidence-first approach:
        1. Collect ALL candidate exons from every GFF source
        2. Score each exon independently (source count, coverage, splice sites)
        3. Use gene models from all GFFs as assembly templates
        4. For each template, find the best-supported exon set
        5. Where templates overlap, keep the best-scoring model
        6. Add genes unique to each source
        """
        consensus = []

        # ================================================================
        # Phase 1: Collect and score all candidate exons
        # ================================================================
        logger.info("  Phase 1: Collecting candidate exons from all sources...")

        # Index: (seqid, strand) -> sorted list of (start, end, sources_set)
        exon_pool_by_region = defaultdict(list)

        def add_exons_from_source(genes, source_label):
            n = 0
            for gene in genes:
                for tx in gene.transcripts:
                    for exon in tx.exons:
                        exon_pool_by_region[(gene.seqid, gene.strand)].append(
                            [exon.start, exon.end, {source_label}])
                        n += 1
            return n

        n_hx = add_exons_from_source(self.helixer_genes, 'Helixer')
        n_td = add_exons_from_source(self.td_genes, 'TransDecoder')
        n_st = add_exons_from_source(self.st_genes, 'StringTie')
        logger.info(f"  Raw exons: Hx={n_hx}, TD={n_td}, ST={n_st}")

        # Merge near-duplicate exons (within tolerance) and combine source tags
        DEDUP_TOL = 10
        candidate_exons = {}  # (seqid, strand, start, end) -> info dict

        for region_key, exon_list in exon_pool_by_region.items():
            seqid, strand = region_key
            exon_list.sort(key=lambda x: (x[0], x[1]))

            merged = []
            for start, end, sources in exon_list:
                placed = False
                # Scan backwards only as far as start values are within DEDUP_TOL.
                # List is sorted by start, so once m[0] < start - DEDUP_TOL we can stop.
                # In practice this scans only a handful of entries per exon → O(n) total.
                for i in range(len(merged) - 1, -1, -1):
                    m = merged[i]
                    if m[0] < start - DEDUP_TOL:
                        break
                    if abs(m[0] - start) <= DEDUP_TOL and abs(m[1] - end) <= DEDUP_TOL:
                        m[2].update(sources)
                        # When two near-duplicate exon calls differ in boundary
                        # position, pick the best position for each end
                        # independently using splice-site evidence: score each
                        # candidate position by (a) how sharply coverage drops
                        # into the flanking intron and (b) the PWM score of the
                        # donor/acceptor sequence at that position.  Work outward
                        # from the smaller (more conservative) exon so that we
                        # only expand a boundary when the evidence supports it.
                        best_start = _pick_best_exon_boundary(
                            seqid, strand, 'start',
                            {m[0], start},
                            self.coverage, self.genome,
                            self.bam_evidence)
                        best_end = _pick_best_exon_boundary(
                            seqid, strand, 'end',
                            {m[1], end},
                            self.coverage, self.genome,
                            self.bam_evidence)
                        m[0], m[1] = best_start, best_end
                        placed = True
                        break
                if not placed:
                    merged.append([start, end, set(sources)])

            for start, end, sources in merged:
                k = (seqid, strand, start, end)
                candidate_exons[k] = {
                    'sources': sources,
                    'start': start,
                    'end': end,
                    'seqid': seqid,
                    'strand': strand,
                }

        logger.info(f"  Candidate exons after dedup: {len(candidate_exons)}")

        # Phase 1 trace: dump candidate exons overlapping any --trace_region
        if self.tracer.enabled and self.tracer.regions:
            for (seqid, strand), _ in exon_pool_by_region.items():
                in_region = sorted(
                    (k[2], k[3], info['sources']) for k, info
                    in candidate_exons.items()
                    if k[0] == seqid and k[1] == strand
                    and self.tracer.region_overlaps(seqid, k[2], k[3]))
                if in_region:
                    logger.info(
                        f"[TRACE] Phase 1 pool | {seqid}({strand}) | "
                        f"{len(in_region)} candidate exons in trace region")
                    for s, e, sources in in_region[:40]:
                        logger.info(f"[TRACE]   {s}-{e} sources={sorted(sources)}")

        # ================================================================
        # Phase 2: Score each candidate exon
        # ================================================================
        logger.info("  Phase 2: Scoring candidate exons...")

        for k, info in candidate_exons.items():
            seqid, strand, start, end = k

            # Count independent source corroboration using the evidence spatial
            # index rather than the pool source tags.  The pool merge can combine
            # exons from nearby but distinct transcripts, falsely inflating the
            # source count.  The evidence_index tracks which prediction tools
            # (Helixer / StringTie / TransDecoder) each have a gene-model exon
            # at this position — these are genuinely independent evidence types.
            independent_sources = set()
            for src_label in ('Helixer', 'StringTie', 'TransDecoder'):
                if self.evidence_index.has_overlapping_exon(
                        seqid, strand, start, end, source=src_label):
                    independent_sources.add(src_label)
            info['independent_sources'] = independent_sources
            n_sources = len(independent_sources)
            info['source_score'] = min(1.0, n_sources / 2.0)

            # RNA-seq coverage
            cov = self.coverage.get_mean_coverage(seqid, start, end)
            info['coverage'] = cov
            info['cov_score'] = min(1.0, cov / 10.0) if cov > 0 else 0.0

            # Coverage ratio (exon vs flanking)
            ratio = self.coverage.get_local_coverage_ratio(seqid, start, end)
            info['ratio_score'] = min(1.0, ratio / 3.0) if ratio > 0 else 0.0

            # Composite exon score
            info['score'] = (0.35 * info['source_score'] +
                            0.35 * info['cov_score'] +
                            0.30 * info['ratio_score'])

        # Build spatial index of scored exons for fast lookup.
        # Tuple layout: (start, end, score, independent_sources, coverage)
        # independent_sources is a frozenset of {'Helixer','StringTie','TransDecoder'}
        # verified via the evidence_index — not inflated by pool-merge accidents.
        scored_exons_by_region = defaultdict(list)
        for k, info in candidate_exons.items():
            seqid, strand, start, end = k
            scored_exons_by_region[(seqid, strand)].append(
                (start, end, info['score'],
                 frozenset(info['independent_sources']),
                 info['coverage']))
        for key in scored_exons_by_region:
            scored_exons_by_region[key].sort()

        def find_matching_exon(seqid, strand, qstart, qend, tol=DEDUP_TOL):
            """Find the scored candidate exon matching a query position."""
            import bisect
            exons = scored_exons_by_region.get((seqid, strand), [])
            idx = bisect.bisect_left(exons, (qstart - tol - 100,))
            best = None
            best_dist = float('inf')
            while idx < len(exons):
                s, e, sc, src, cov = exons[idx]
                if s > qstart + tol + 100:
                    break
                d = abs(s - qstart) + abs(e - qend)
                if d <= tol * 2 and d < best_dist:
                    best = (s, e, sc, src, cov)
                    best_dist = d
                idx += 1
            return best

        # ================================================================
        # Phase 3: Assemble genes — start from templates, then modify
        # ================================================================
        logger.info("  Phase 3: Assembling genes from templates + evidence...")

        # Collect all gene templates from all sources
        all_templates = []
        for gene in self.helixer_genes:
            all_templates.append(('Helixer', gene))
        for gene in self.td_genes:
            all_templates.append(('TransDecoder', gene))
        for gene in self.st_genes:
            all_templates.append(('StringTie', gene))

        assembled_genes = []  # (mean_score, Gene)

        for source_label, template in all_templates:
            if not template.transcripts:
                continue

            # Select the best-supported isoform rather than the longest.
            # Score each isoform by:
            #   (1) fewest non-canonical introns          — primary key
            #   (2) number of junction-confirmed introns  — secondary key
            #   (3) mean exon coverage                    — tiebreaker
            # Non-canonical introns come first because a TransDecoder isoform
            # can pick up an extra junction-confirmed intron that traces back
            # to a misassembled transcript (e.g. 020810's g3005661.31 has a
            # CA-AG intron that would otherwise win on junction count alone).
            # Ranking canonical isoforms first rejects those chimeras before
            # they enter Phase 4.
            def _score_isoform(tx):
                n_junc = 0
                n_noncanon = 0
                for intron_s, intron_e in tx.introns():
                    if self.bam_evidence.count_spliced_reads(
                            template.seqid, intron_s, intron_e) > 0:
                        n_junc += 1
                    if not _intron_is_canonical(
                            template.seqid, template.strand,
                            intron_s, intron_e, self.genome):
                        n_noncanon += 1
                if tx.exons:
                    mean_cov = sum(
                        self.coverage.get_mean_coverage(
                            template.seqid, e.start, e.end)
                        for e in tx.exons
                    ) / len(tx.exons)
                else:
                    mean_cov = 0.0
                return (-n_noncanon, n_junc, mean_cov)

            best_tx = max(template.transcripts, key=_score_isoform)
            if not best_tx.exons:
                continue

            seqid = template.seqid
            strand = template.strand

            # Template quality filter: skip multi-exon templates below the
            # empirically-derived minimum junction support threshold.
            # The threshold (1st percentile of mean-reads-per-intron across
            # multi-exon StringTie transcripts) gives a non-zero, depth-scaled
            # floor that rejects noise transcripts like STRG.12310 whose
            # introns have zero junction reads.
            # Single-exon templates are always accepted (no introns to check).
            #
            # TransDecoder templates are EXEMPT from this filter.  Their exon
            # structure comes from actual transcript sequences (isoseq or
            # assembled transcriptomes), not from an RNA-seq assembler.  The
            # portcullis junction file is derived from short-read alignments
            # and may not confirm isoseq-derived introns even when the splice
            # sites are real (e.g. different mapping tolerances, tissue
            # specificity, or low coverage in the short-read library).
            if self.bam_evidence.available and source_label not in ('TransDecoder', 'Manual'):
                introns = list(best_tx.introns())
                if introns:
                    counts = [
                        self.bam_evidence.count_spliced_reads(seqid, s, e)
                        for s, e in introns if e > s]
                    mean_junc = sum(counts) / len(counts) if counts else 0.0
                    if mean_junc < self.calibrator.template_min_junction_mean:
                        logger.debug(
                            f"Skipping template {template.gene_id} "
                            f"({source_label}): mean junction reads "
                            f"{mean_junc:.2f} < threshold "
                            f"{self.calibrator.template_min_junction_mean:.2f}")
                        continue

            # --- Step A: Start with template exons, scored from pool ---
            gene_exons = []
            gene_sources = set()
            total_score = 0.0

            for texon in sorted(best_tx.exons, key=lambda e: e.start):
                match = find_matching_exon(seqid, strand, texon.start, texon.end)
                if match:
                    s, e, sc, src, cov = match
                    # src is already a frozenset of independent tool sources
                    gene_exons.append(Feature(
                        seqid=seqid, source='Refined', ftype='exon',
                        start=s, end=e, score=sc, strand=strand,
                        phase='.', attributes={
                            'sources': ','.join(sorted(src))
                        }
                    ))
                    gene_sources.update(src)
                    total_score += sc
                else:
                    # Template exon not in pool (shouldn't happen, but safety)
                    gene_exons.append(Feature(
                        seqid=seqid, source='Refined', ftype='exon',
                        start=texon.start, end=texon.end, score=0.1,
                        strand=strand, phase='.',
                        attributes={'sources': source_label}
                    ))
                    gene_sources.add(source_label)
                    total_score += 0.1

            if not gene_exons:
                continue

            # --- Step B: Drop poorly-supported internal exons ---
            # Each internal exon is scored with exon_posterior (calibrated from
            # the StringTie coverage-ratio and junction-count distributions).
            # Exons below the data-derived drop threshold are dropped if their
            # removal would produce a canonical GT-AG (or GC-AG) intron.
            if len(gene_exons) >= 3:
                sorted_ge = sorted(gene_exons, key=lambda e: e.start)
                keep = [True] * len(sorted_ge)

                # Per-exon mean coverage
                exon_coverages = [
                    self.coverage.get_mean_coverage(seqid, e.start, e.end)
                    for e in sorted_ge
                ]
                # Gene-level median coverage (for normalisation)
                non_zero_covs = [c for c in exon_coverages if c > 0]
                gene_median_cov = (sorted(non_zero_covs)[len(non_zero_covs) // 2]
                                   if non_zero_covs else 0.0)

                for i in range(1, len(sorted_ge) - 1):
                    exon     = sorted_ge[i]
                    exon_cov = exon_coverages[i]
                    src_set  = set(exon.attributes.get('sources', '').split(','))

                    # Flanking introns for junction-count lookup
                    left_intron  = (sorted_ge[i - 1].end + 1, exon.start - 1)
                    right_intron = (exon.end + 1, sorted_ge[i + 1].start - 1)

                    posterior = self.calibrator.score_exon(
                        seqid, exon.start, exon.end,
                        gene_median_cov,
                        [left_intron, right_intron],
                        n_sources=len(src_set))

                    # Hard override: if junction evidence is available and
                    # BOTH flanking introns have zero reads, force posterior
                    # to 0 regardless of coverage or source count.  An exon
                    # whose neighbouring splice sites have no RNA-seq support
                    # at all is not a real internal exon — coverage alone is
                    # not sufficient evidence.  Sequencing errors or rare
                    # alternative sites can produce canonical GT-AG dinucleotides
                    # at non-functional positions; only junction reads confirm use.
                    #
                    # Exception: TransDecoder support exempts an exon from this
                    # override.  TransDecoder CDS comes from actual transcript
                    # sequences (isoseq or assembled), so its exon structure is
                    # real even when portcullis does not confirm the junctions
                    # (e.g. because the gene is lowly expressed and portcullis
                    # filtered out low-count junctions in its reliability pass).
                    has_trusted_support = bool(src_set & {'TransDecoder', 'Manual'})
                    if self.bam_evidence.available and not has_trusted_support:
                        left_reads  = self.bam_evidence.count_spliced_reads(
                            seqid, left_intron[0],  left_intron[1])
                        right_reads = self.bam_evidence.count_spliced_reads(
                            seqid, right_intron[0], right_intron[1])
                        if left_reads == 0 and right_reads == 0:
                            posterior = 0.0

                    if posterior < self.calibrator.drop_threshold:
                        # Only drop if removal creates a canonical intron
                        gap_s = sorted_ge[i - 1].end + 1
                        gap_e = sorted_ge[i + 1].start - 1
                        if gap_e - gap_s + 1 >= MIN_INTRON_SIZE:
                            if strand == '+':
                                d = self.genome.get_sequence(seqid, gap_s, gap_s + 1).upper()
                                a = self.genome.get_sequence(seqid, gap_e - 1, gap_e).upper()
                            else:
                                d = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_e - 1, gap_e)).upper()
                                a = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_s, gap_s + 1)).upper()
                            if len(d) >= 2 and len(a) >= 2:
                                if (d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG'):
                                    keep[i] = False
                                    total_score -= exon.score
                                    logger.debug(
                                        f"Dropping exon {exon.start}-{exon.end} "
                                        f"(posterior={posterior:.3f} < "
                                        f"threshold={self.calibrator.drop_threshold:.3f}, "
                                        f"cov={exon_cov:.1f}, "
                                        f"gene_median={gene_median_cov:.1f}, "
                                        f"sources={src_set})")

                gene_exons = [e for e, k in zip(sorted_ge, keep) if k]

            # --- Step C: Add well-supported exons from other sources ---
            # Look for exons in the candidate pool that:
            # - Fall within or adjacent to the gene span
            # - Are supported by 2+ sources OR have coverage consistent with neighbors
            # - Don't overlap existing exons
            # - Create canonical splice sites with neighbors
            if gene_exons:
                gene_start = min(e.start for e in gene_exons)
                gene_end = max(e.end for e in gene_exons)

                # Compute median coverage of existing gene exons for reference
                existing_covs = [self.coverage.get_mean_coverage(seqid, e.start, e.end)
                                 for e in gene_exons]
                median_gene_cov = sorted(existing_covs)[len(existing_covs) // 2] if existing_covs else 0.0

                # Search window: gene span ± 5kb for potential UTR exons
                search_start = max(1, gene_start - 5000)
                search_end = gene_end + 5000

                pool_exons = scored_exons_by_region.get((seqid, strand), [])
                import bisect
                idx = bisect.bisect_left(pool_exons, (search_start,))

                additions = []
                while idx < len(pool_exons):
                    ps, pe, psc, psrc, pcov = pool_exons[idx]
                    if ps > search_end:
                        break
                    idx += 1

                    # Skip if already in gene model
                    already = any(abs(e.start - ps) <= DEDUP_TOL and abs(e.end - pe) <= DEDUP_TOL
                                 for e in gene_exons)
                    if already:
                        continue

                    # Skip if overlaps an existing exon
                    overlaps = any(ps <= e.end and pe >= e.start for e in gene_exons)
                    if overlaps:
                        continue

                    # Require calibrated posterior above drop_threshold.
                    # psrc is a frozenset of independently verified source tools.
                    # We use the current gene median (median_gene_cov) for
                    # normalisation so the threshold is expression-level agnostic.
                    # Flanking introns are unknown at this point (exon not yet
                    # inserted), so we pass empty flanking and let junction
                    # evidence be scored properly in the posterior annotation pass.
                    candidate_posterior = self.calibrator.score_exon(
                        seqid, ps, pe,
                        median_gene_cov,
                        flanking_introns=[],
                        n_sources=len(psrc))
                    if candidate_posterior < self.calibrator.drop_threshold:
                        continue

                    # Check splice site compatibility with nearest existing exons
                    sorted_ge = sorted(gene_exons + additions, key=lambda e: e.start)
                    # Find where this exon would be inserted
                    insert_pos = 0
                    for j, e in enumerate(sorted_ge):
                        if e.start > ps:
                            break
                        insert_pos = j + 1

                    # Check splice sites with left neighbor
                    left_ok = True
                    if insert_pos > 0:
                        left_e = sorted_ge[insert_pos - 1]
                        gap_s = left_e.end + 1
                        gap_e = ps - 1
                        if gap_e - gap_s + 1 >= MIN_INTRON_SIZE:
                            if strand == '+':
                                d = self.genome.get_sequence(seqid, gap_s, gap_s + 1).upper()
                                a = self.genome.get_sequence(seqid, gap_e - 1, gap_e).upper()
                            else:
                                d = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_e - 1, gap_e)).upper()
                                a = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_s, gap_s + 1)).upper()
                            if len(d) >= 2 and len(a) >= 2:
                                if not ((d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG')):
                                    left_ok = False
                        elif gap_e - gap_s + 1 > 0:
                            left_ok = False  # too-short intron

                    # Check splice sites with right neighbor
                    right_ok = True
                    if insert_pos < len(sorted_ge):
                        right_e = sorted_ge[insert_pos]
                        gap_s = pe + 1
                        gap_e = right_e.start - 1
                        if gap_e - gap_s + 1 >= MIN_INTRON_SIZE:
                            if strand == '+':
                                d = self.genome.get_sequence(seqid, gap_s, gap_s + 1).upper()
                                a = self.genome.get_sequence(seqid, gap_e - 1, gap_e).upper()
                            else:
                                d = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_e - 1, gap_e)).upper()
                                a = reverse_complement(
                                    self.genome.get_sequence(seqid, gap_s, gap_s + 1)).upper()
                            if len(d) >= 2 and len(a) >= 2:
                                if not ((d == 'GT' and a == 'AG') or (d == 'GC' and a == 'AG')):
                                    right_ok = False
                        elif gap_e - gap_s + 1 > 0:
                            right_ok = False

                    # For terminal positions (would be first or last exon),
                    # require junction support on the inner (gene-body-facing)
                    # intron.  Without this, exons from nearby unrelated
                    # transcripts get pulled in through the ±5kb search window.
                    if left_ok and right_ok:
                        is_terminal = (insert_pos == 0 or
                                       insert_pos == len(sorted_ge))
                        if is_terminal and self.bam_evidence.available:
                            if insert_pos == 0 and len(sorted_ge) > 0:
                                # Would be new first exon: check junction to right neighbour
                                inner_s = pe + 1
                                inner_e = sorted_ge[0].start - 1
                            elif insert_pos == len(sorted_ge) and len(sorted_ge) > 0:
                                # Would be new last exon: check junction from left neighbour
                                inner_s = sorted_ge[-1].end + 1
                                inner_e = ps - 1
                            else:
                                inner_s = inner_e = 0
                            if (inner_e > inner_s and
                                    self.bam_evidence.count_spliced_reads(
                                        seqid, inner_s, inner_e) == 0):
                                logger.debug(
                                    f"Rejecting terminal exon {ps}-{pe}: "
                                    f"no junction support on inner intron "
                                    f"{inner_s}-{inner_e}")
                                continue

                        new_exon = Feature(
                            seqid=seqid, source='Refined', ftype='exon',
                            start=ps, end=pe, score=psc, strand=strand,
                            phase='.', attributes={
                                'sources': ','.join(sorted(psrc)),
                                'added_from_pool': 'true'
                            }
                        )
                        additions.append(new_exon)
                        gene_sources.update(psrc)
                        total_score += psc
                        logger.debug(f"Adding exon {ps}-{pe} to gene "
                                   f"(score={psc:.2f}, cov={pcov:.1f}, sources={psrc})")

                gene_exons.extend(additions)
                gene_exons.sort(key=lambda e: e.start)

            if not gene_exons:
                continue

            # --- Posterior annotation pass ---
            # Compute exon_posterior for every retained exon (including any
            # added in Step C) using the calibrated distributions, and store
            # it as an attribute so it propagates to the GFF output.
            sorted_final = sorted(gene_exons, key=lambda e: e.start)
            final_covs = [self.coverage.get_mean_coverage(seqid, e.start, e.end)
                          for e in sorted_final]
            nz = [c for c in final_covs if c > 0]
            final_gene_median = sorted(nz)[len(nz) // 2] if nz else 0.0
            for i, exon in enumerate(sorted_final):
                flanking = []
                if i > 0:
                    flanking.append((sorted_final[i - 1].end + 1, exon.start - 1))
                if i < len(sorted_final) - 1:
                    flanking.append((exon.end + 1, sorted_final[i + 1].start - 1))
                n_src = len(set(exon.attributes.get('sources', '').split(',')))
                post = self.calibrator.score_exon(
                    seqid, exon.start, exon.end,
                    final_gene_median, flanking, n_src)
                exon.attributes['exon_posterior'] = f'{post:.4f}'

            # --- Terminal exon junction audit ---
            # Working inward from each end of the assembled transcript, drop
            # terminal exons whose connecting intron has zero junction reads.
            # Also drop terminal exons where the connecting intron has very
            # few reads (< TERMINAL_MIN_READS) AND the terminal exon coverage
            # is much lower than the adjacent anchor exon (< TERMINAL_COV_RATIO).
            # This removes rare-alternative-splice terminal exons that would
            # otherwise beat the dominant high-coverage UTR-extended isoform
            # in Phase 4 merely because they contribute one extra confirmed intron.
            # Stop at the first exon that passes both checks.
            TERMINAL_MIN_READS = 5      # below this, also check coverage ratio
            TERMINAL_COV_RATIO = 0.10   # terminal exon must be > 10% of anchor cov
            TERMINAL_EXTREME_COV_RATIO = 0.01  # even with junctions, drop if < 1%

            def _low_confidence_terminal(terminal_exon, anchor_exon, intron_reads):
                """Return True if the terminal exon looks like a rare minor isoform."""
                if intron_reads < 1:
                    return True
                t_cov = self.coverage.get_mean_coverage(
                    seqid, terminal_exon.start, terminal_exon.end)
                a_cov = self.coverage.get_mean_coverage(
                    seqid, anchor_exon.start, anchor_exon.end)
                if intron_reads < TERMINAL_MIN_READS:
                    if a_cov > 1.0 and t_cov < a_cov * TERMINAL_COV_RATIO:
                        return True
                # Even with many junction reads, drop terminal exons with
                # extreme coverage disparity relative to the anchor — these
                # are typically from a different gene or a rare minor isoform
                # that happens to share a splice site.
                if a_cov > 1.0 and t_cov < a_cov * TERMINAL_EXTREME_COV_RATIO:
                    n_src = len(set(
                        terminal_exon.attributes.get('sources', '').split(',')))
                    if n_src <= 1:
                        return True
                return False

            if gene_exons and self.bam_evidence.available:
                _tx_order = sorted(gene_exons, key=lambda e: e.start)

                # Trim from 5' end (low index on + strand)
                while len(_tx_order) > 1:
                    _is = _tx_order[0].end + 1
                    _ie = _tx_order[1].start - 1
                    if _ie > _is:
                        _reads = self.bam_evidence.count_spliced_reads(seqid, _is, _ie)
                        if _low_confidence_terminal(_tx_order[0], _tx_order[1], _reads):
                            logger.debug(
                                f"  Terminal audit: dropping 5'-exon "
                                f"{_tx_order[0].start}-{_tx_order[0].end} "
                                f"({template.gene_id}, intron {_is}-{_ie}: "
                                f"{_reads} jct reads)")
                            _tx_order.pop(0)
                            continue
                    break

                # Trim from 3' end (high index)
                while len(_tx_order) > 1:
                    _is = _tx_order[-2].end + 1
                    _ie = _tx_order[-1].start - 1
                    if _ie > _is:
                        _reads = self.bam_evidence.count_spliced_reads(seqid, _is, _ie)
                        if _low_confidence_terminal(_tx_order[-1], _tx_order[-2], _reads):
                            logger.debug(
                                f"  Terminal audit: dropping 3'-exon "
                                f"{_tx_order[-1].start}-{_tx_order[-1].end} "
                                f"({template.gene_id}, intron {_is}-{_ie}: "
                                f"{_reads} jct reads)")
                            _tx_order.pop()
                            continue
                    break

                gene_exons = _tx_order

            # Use mean calibrator posterior (not pool score) for Phase 4
            # template selection, so assemblies with well-supported exons
            # (good junction + coverage evidence) beat Helixer-dominated
            # models whose pool scores are inflated by source-count alone.
            posteriors = [float(e.attributes.get('exon_posterior', 0.5))
                          for e in gene_exons]
            mean_score = sum(posteriors) / len(posteriors) if posteriors else 0.0

            # Build gene model
            gene_start = min(e.start for e in gene_exons)
            gene_end = max(e.end for e in gene_exons)

            cgene = Gene(
                gene_id=f"consensus_{source_label}_{template.gene_id}",
                seqid=seqid,
                strand=strand,
                start=gene_start,
                end=gene_end,
                source='Refined',
                attributes={
                    'ID': f"consensus_{source_label}_{template.gene_id}",
                    'template_source': source_label,
                    'template_id': template.gene_id,
                    'evidence_sources': ','.join(sorted(gene_sources))
                }
            )

            ctx = Transcript(
                transcript_id=f"{cgene.gene_id}.1",
                seqid=seqid, strand=strand,
                start=gene_start, end=gene_end,
                source='Refined'
            )
            ctx.exons = gene_exons

            # ---- CDS assignment ----
            # If the template has a CDS that spans most of the transcript
            # (starts in the first 3 exons AND ends in the last 3 exons),
            # recalculate the CDS from scratch using the refined exon set so
            # that (a) the start codon is the first ATG in the correct exon,
            # and (b) the CDS coordinates match the potentially shifted exon
            # boundaries.  This corrects cases where the template CDS was
            # built from a different exon structure.
            #
            # If the CDS is partial / internal (e.g. just a CDS fragment that
            # does not span most of the transcript), carry it over unchanged —
            # recalculating from scratch could produce a different ORF.
            if best_tx.cds and gene_exons:
                sorted_exons = sorted(gene_exons, key=lambda e: e.start)
                n_exons = len(sorted_exons)
                cds_sorted = sorted(best_tx.cds, key=lambda c: c.start)
                cds_min = cds_sorted[0].start
                cds_max = cds_sorted[-1].end

                # Find which exon index (0-based) contains the CDS start/end
                # in the *refined* exon set (which may differ from the template
                # if terminal exons were trimmed by the junction audit above).
                cds_start_exon_idx = next(
                    (i for i, e in enumerate(sorted_exons)
                     if e.start <= cds_min <= e.end), None)
                cds_end_exon_idx = next(
                    (i for i, e in enumerate(sorted_exons)
                     if e.start <= cds_max <= e.end), None)

                # Recalculate CDS if:
                # (a) CDS spans most of the refined transcript (starts in first
                #     3 exons AND ends in last 3 exons), OR
                # (b) The CDS start is beyond exon 3 — this happens when
                #     terminal-exon trimming removed leading exons that were
                #     upstream of the real coding region, leaving the CDS now
                #     orphaned deep in the transcript.  Recalculating from the
                #     first ATG will fix the frame.
                last3_start = max(0, n_exons - 3)
                cds_starts_early = (cds_start_exon_idx is not None and
                                    cds_start_exon_idx < 3)
                cds_ends_late    = (cds_end_exon_idx is not None and
                                    cds_end_exon_idx >= last3_start)
                cds_starts_late  = (cds_start_exon_idx is not None and
                                    cds_start_exon_idx >= 3)

                if (cds_starts_early and cds_ends_late) or cds_starts_late:
                    orf_finder = ORFFinder(self.genome)
                    ev_starts = self.evidence_index.get_evidence_cds_starts(
                        seqid, strand, cgene.start, cgene.end)
                    orf = orf_finder.find_best_orf(seqid, gene_exons, strand,
                                                   coverage=self.coverage,
                                                   evidence_cds_starts=ev_starts)
                    if orf:
                        orf_start, orf_end, _ = orf
                        new_cds = orf_finder.orf_to_genomic_cds(
                            seqid, gene_exons, strand, orf_start, orf_end)
                        if new_cds:
                            ctx.cds = new_cds
                            logger.debug(
                                f"Recalculated CDS for {cgene.gene_id} "
                                f"from first ATG: "
                                f"{new_cds[0].start}-{new_cds[-1].end}")
                        else:
                            ctx.cds = list(best_tx.cds)
                    else:
                        ctx.cds = list(best_tx.cds)
                else:
                    # Partial / internal CDS with correct position — carry over
                    ctx.cds = list(best_tx.cds)

            if best_tx.five_prime_utrs:
                ctx.five_prime_utrs = list(best_tx.five_prime_utrs)
            if best_tx.three_prime_utrs:
                ctx.three_prime_utrs = list(best_tx.three_prime_utrs)

            # Count junction-supported introns for Phase 4 sorting.
            # This is the PRIMARY sort key: a model with more confirmed
            # introns is always preferred over a fragment with fewer,
            # regardless of mean posterior.  This prevents Helixer fragments
            # (high mean posterior over their small sub-region) from blocking
            # a full-length TransDecoder model (lower mean over 100+ exons).
            n_junc_supported = 0
            for intron_start, intron_end in ctx.introns():
                if self.bam_evidence.count_spliced_reads(
                        seqid, intron_start, intron_end) > 0:
                    n_junc_supported += 1
            cgene.attributes['_n_junc_supported'] = n_junc_supported

            cgene.transcripts.append(ctx)
            assembled_genes.append((mean_score, cgene))

        logger.info(f"  Assembled {len(assembled_genes)} template-based gene models")

        # Phase 3 trace: show candidate assembled models for target loci
        if self.tracer.enabled:
            for score, g in assembled_genes:
                if self.tracer.matches(g):
                    logger.info(
                        f"[TRACE] Phase 3 candidate | "
                        f"src={g.attributes.get('evidence_sources', '?')} "
                        f"score={score:.3f} "
                        f"junc={g.attributes.get('_n_junc_supported', 0)}")
                    self.tracer.snapshot("Phase 3 candidate", [g])

        # ================================================================
        # Phase 4: Resolve overlapping templates — keep the best
        # ================================================================
        logger.info("  Phase 4: Resolving overlapping models...")

        # Sort primarily by number of junction-confirmed introns (descending),
        # then by mean exon posterior (descending) as a tiebreaker.
        # Rationale: a model whose intron structure is confirmed by more
        # independent splice-junction reads is more trustworthy than one
        # whose exons merely look good in isolation.  This ensures a
        # full-length TransDecoder model (e.g. 100 confirmed introns) beats
        # six Helixer fragments (each ~20 confirmed introns) that together
        # cover the same locus.
        assembled_genes.sort(key=lambda x: (
            -x[1].attributes.get('_n_junc_supported', 0),
            -x[0]  # mean_score (mean exon posterior) as tiebreaker
        ))

        # Greedy selection: best-scored gene wins at each locus
        selected = []
        occupied = defaultdict(list)  # (seqid, strand) -> [(start, end)]

        for _, gene in assembled_genes:
            key = (gene.seqid, gene.strand)
            overlaps_selected = False

            for (ostart, oend) in occupied[key]:
                overlap = min(gene.end, oend) - max(gene.start, ostart)
                if overlap > 0:
                    min_len = min(gene.end - gene.start, oend - ostart)
                    if min_len > 0 and overlap / min_len > 0.3:
                        overlaps_selected = True
                        break

            if not overlaps_selected:
                selected.append(gene)
                occupied[key].append((gene.start, gene.end))

        logger.info(f"  Selected {len(selected)} non-overlapping gene models "
                   f"(from {len(assembled_genes)} candidates)")
        self.tracer.snapshot("After Phase 4 (selected)", selected)

        # ================================================================
        # Phase 4.5: Extend terminal exons to recover UTRs
        # ================================================================
        # Phase 3 picks the winning template's exon boundaries verbatim.  If
        # a StringTie template beats a Helixer/TD alternative by a small
        # posterior margin but lacks UTR extent, the refined gene inherits
        # the narrower StringTie boundaries and loses the UTR that Helixer
        # or TransDecoder would have supplied.
        # Recover UTR by extending each terminal exon's outward boundary to
        # the widest Helixer/TD pool boundary that (a) shares the inward
        # splice site within tolerance and (b) has RNA-seq coverage across
        # the extension consistent with transcription.
        logger.info("  Phase 4.5: Extending terminal exons using UTR pool...")
        TERMINAL_INWARD_TOL = 10
        UTR_COV_ABS_MIN = 1.0
        UTR_COV_REL_MIN = 0.20
        UTR_TRUSTED_SOURCES = frozenset({'Helixer', 'TransDecoder'})

        selected_footprints = defaultdict(list)
        for other in selected:
            selected_footprints[(other.seqid, other.strand)].append(
                (other.gene_id, other.start, other.end))

        n_extended = 0
        for gene in selected:
            pool = scored_exons_by_region.get((gene.seqid, gene.strand), [])
            if not pool:
                continue
            others = [(gid, s, e) for (gid, s, e)
                      in selected_footprints[(gene.seqid, gene.strand)]
                      if gid != gene.gene_id]

            for tx in gene.transcripts:
                if not tx.exons:
                    continue
                sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                multi_exon = len(sorted_exons) > 1

                # --- Extend the low-coordinate terminal exon outward (lower) ---
                first_exon = sorted_exons[0]
                original_start = first_exon.start
                best_start = original_start
                exon_cov_first = self.coverage.get_mean_coverage(
                    gene.seqid, first_exon.start, first_exon.end)

                for pstart, pend, _sc, psrc, _pcov in pool:
                    if not (psrc & UTR_TRUSTED_SOURCES):
                        continue
                    if pstart >= best_start:
                        continue
                    if multi_exon:
                        if abs(pend - first_exon.end) > TERMINAL_INWARD_TOL:
                            continue
                    else:
                        ovl = min(first_exon.end, pend) - max(first_exon.start, pstart)
                        if ovl < 0.5 * (first_exon.end - first_exon.start + 1):
                            continue
                    ext_start, ext_end = pstart, original_start - 1
                    if ext_end < ext_start:
                        continue
                    ext_cov = self.coverage.get_mean_coverage(
                        gene.seqid, ext_start, ext_end)
                    if ext_cov < UTR_COV_ABS_MIN:
                        continue
                    if exon_cov_first > 0 and ext_cov < UTR_COV_REL_MIN * exon_cov_first:
                        continue
                    if any(os_ <= ext_end and oe_ >= ext_start
                           for (_g, os_, oe_) in others):
                        continue
                    if not self._passes_strand_check(
                            gene.seqid, ext_start, ext_end, gene.strand, ext_cov):
                        if self.tracer.enabled and self.tracer.matches(gene):
                            logger.info(
                                f"[TRACE] Phase 4.5 reject low-end (antisense) | "
                                f"{gene.gene_id} | candidate {pstart}-{original_start - 1}")
                        continue
                    best_start = pstart

                if best_start < original_start:
                    if self.tracer.enabled and self.tracer.matches(gene):
                        logger.info(
                            f"[TRACE] Phase 4.5 extend low-end | {gene.gene_id} | "
                            f"{original_start} -> {best_start} "
                            f"(+{original_start - best_start} bp)")
                    first_exon.start = best_start
                    n_extended += 1

                # --- Extend the high-coordinate terminal exon outward (higher) ---
                last_exon = sorted_exons[-1]
                original_end = last_exon.end
                best_end = original_end
                exon_cov_last = self.coverage.get_mean_coverage(
                    gene.seqid, last_exon.start, last_exon.end)

                for pstart, pend, _sc, psrc, _pcov in pool:
                    if not (psrc & UTR_TRUSTED_SOURCES):
                        continue
                    if pend <= best_end:
                        continue
                    if multi_exon:
                        if abs(pstart - last_exon.start) > TERMINAL_INWARD_TOL:
                            continue
                    else:
                        ovl = min(last_exon.end, pend) - max(last_exon.start, pstart)
                        if ovl < 0.5 * (last_exon.end - last_exon.start + 1):
                            continue
                    ext_start, ext_end = original_end + 1, pend
                    if ext_end < ext_start:
                        continue
                    ext_cov = self.coverage.get_mean_coverage(
                        gene.seqid, ext_start, ext_end)
                    if ext_cov < UTR_COV_ABS_MIN:
                        continue
                    if exon_cov_last > 0 and ext_cov < UTR_COV_REL_MIN * exon_cov_last:
                        continue
                    if any(os_ <= ext_end and oe_ >= ext_start
                           for (_g, os_, oe_) in others):
                        continue
                    if not self._passes_strand_check(
                            gene.seqid, ext_start, ext_end, gene.strand, ext_cov):
                        if self.tracer.enabled and self.tracer.matches(gene):
                            logger.info(
                                f"[TRACE] Phase 4.5 reject high-end (antisense) | "
                                f"{gene.gene_id} | candidate {original_end + 1}-{pend}")
                        continue
                    best_end = pend

                if best_end > original_end:
                    if self.tracer.enabled and self.tracer.matches(gene):
                        logger.info(
                            f"[TRACE] Phase 4.5 extend high-end | {gene.gene_id} | "
                            f"{original_end} -> {best_end} "
                            f"(+{best_end - original_end} bp)")
                    last_exon.end = best_end
                    n_extended += 1

            self._recompute_gene_boundaries(gene)

        if n_extended:
            logger.info(f"  Phase 4.5: Extended {n_extended} terminal exon boundary(ies)")
        self.tracer.snapshot("After Phase 4.5 (UTR extension)", selected)

        # ================================================================
        # Phase 5: Trim zero-coverage terminal exons
        # ================================================================
        for gene in selected:
            for tx in gene.transcripts:
                tx = trim_zero_coverage_terminal_exons(
                    tx, self.coverage, gene.seqid, gene.strand)

        consensus = selected
        self.tracer.snapshot("After Phase 5 (terminal exon trim)", consensus)
        return consensus

    @staticmethod
    def _bridge_merge_helixer(helixer_genes: List[Gene],
                               td_genes: List[Gene]) -> List[List[Gene]]:
        """Group Helixer genes that should be merged based on TransDecoder bridging.

        If a single TransDecoder gene overlaps multiple Helixer genes on the same
        strand, those Helixer genes are grouped together.
        """
        from collections import defaultdict

        # Index Helixer genes
        h_by_strand = defaultdict(list)
        for hg in helixer_genes:
            h_by_strand[(hg.seqid, hg.strand)].append(hg)

        # Build union-find for merging
        parent = {hg.gene_id: hg.gene_id for hg in helixer_genes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Check each TD gene for bridging
        for tdg in td_genes:
            key = (tdg.seqid, tdg.strand)
            overlapping = [
                hg for hg in h_by_strand.get(key, [])
                if hg.start <= tdg.end and hg.end >= tdg.start
            ]
            if len(overlapping) > 1:
                # Bridge: union all overlapping Helixer genes
                for i in range(1, len(overlapping)):
                    union(overlapping[0].gene_id, overlapping[i].gene_id)
                    logger.info(f"Bridge-merging Helixer {overlapping[0].gene_id} + "
                               f"{overlapping[i].gene_id} (bridged by TD {tdg.gene_id})")

        # Also merge adjacent Helixer genes with gap <= 2bp (likely split errors)
        for key, hlist in h_by_strand.items():
            hlist_sorted = sorted(hlist, key=lambda g: g.start)
            for i in range(len(hlist_sorted) - 1):
                gap = hlist_sorted[i + 1].start - hlist_sorted[i].end - 1
                if gap <= 2:
                    union(hlist_sorted[i].gene_id, hlist_sorted[i + 1].gene_id)
                    logger.info(f"Adjacent-merging Helixer {hlist_sorted[i].gene_id} + "
                               f"{hlist_sorted[i + 1].gene_id} (gap={gap}bp)")

        # Group by root
        groups = defaultdict(list)
        for hg in helixer_genes:
            groups[find(hg.gene_id)].append(hg)

        # Sort each group by position
        result = []
        for glist in groups.values():
            glist.sort(key=lambda g: g.start)
            result.append(glist)

        # Sort groups by first gene's start
        result.sort(key=lambda gl: gl[0].start)
        return result

    def _overlap_score(self, gene_a: Gene, gene_b: Gene) -> float:
        """Calculate overlap score between two genes."""
        overlap_start = max(gene_a.start, gene_b.start)
        overlap_end = min(gene_a.end, gene_b.end)
        if overlap_end < overlap_start:
            return 0.0
        overlap = overlap_end - overlap_start + 1
        min_len = min(gene_a.end - gene_a.start + 1, gene_b.end - gene_b.start + 1)
        return overlap / max(min_len, 1)

    def _refine_exons_with_stringtie(self, tx: Transcript,
                                      st_genes: List[Gene]) -> Transcript:
        """Refine exon boundaries using StringTie evidence.

        StringTie exon boundaries are based on actual RNA-seq reads, so they
        may be more accurate than Helixer predictions for exact splice sites.
        """
        # Collect all StringTie exons
        st_exons = []
        for sg in st_genes:
            for stx in sg.transcripts:
                st_exons.extend(stx.exons)

        if not st_exons:
            return tx

        # For each exon in the consensus, check if StringTie gives better boundaries
        refined_exons = []
        for exon in tx.sorted_exons():
            best_match = None
            best_overlap = 0

            for st_exon in st_exons:
                overlap_start = max(exon.start, st_exon.start)
                overlap_end = min(exon.end, st_exon.end)
                if overlap_end >= overlap_start:
                    overlap = overlap_end - overlap_start + 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = st_exon

            if best_match and best_overlap > 0:
                # Use StringTie boundaries if they're close to Helixer
                # but only adjust by small amounts to avoid major structural changes
                new_start = exon.start
                new_end = exon.end

                if abs(best_match.start - exon.start) < 50:
                    new_start = best_match.start
                if abs(best_match.end - exon.end) < 50:
                    new_end = best_match.end

                refined_exon = Feature(
                    seqid=exon.seqid, source='Refined', ftype='exon',
                    start=new_start, end=new_end, score=exon.score,
                    strand=exon.strand, phase=exon.phase, attributes=exon.attributes
                )
                refined_exons.append(refined_exon)
            else:
                refined_exons.append(exon)

        tx.exons = refined_exons
        return tx

    def _split_by_stringtie(self, gene: Gene) -> List[Gene]:
        """Split a gene ONLY when ALL conditions are met:
        1. Multiple non-overlapping ST gene clusters exist
        2. Separate TD genes exist for each side
        3. No TD gene spans across the gap
        4. No well-covered exons exist in the gap between clusters
        """
        overlapping_st = [
            g for g in self.st_genes
            if g.seqid == gene.seqid
            and (g.strand == gene.strand or g.strand == '.')
            and g.start <= gene.end and g.end >= gene.start
        ]
        overlapping_td = [
            g for g in self.td_genes
            if g.seqid == gene.seqid and g.strand == gene.strand
            and g.start <= gene.end and g.end >= gene.start
        ]

        if len(overlapping_st) < 2 or len(overlapping_td) < 2:
            return [gene]

        # Find ST clusters separated by >500bp gaps
        overlapping_st.sort(key=lambda g: g.start)
        clusters = [[overlapping_st[0]]]
        for st in overlapping_st[1:]:
            prev_end = max(g.end for g in clusters[-1])
            if st.start > prev_end + 500:
                clusters.append([st])
            else:
                clusters[-1].append(st)

        if len(clusters) < 2:
            return [gene]

        # Verify no shared ST gene_id across clusters
        cluster_ids = [set(g.gene_id for g in cl) for cl in clusters]
        if set.intersection(*cluster_ids):
            return [gene]

        # Verify no TD gene spans across any cluster gap
        td_sorted = sorted(overlapping_td, key=lambda g: g.start)
        for ci in range(len(clusters) - 1):
            gap_start = max(g.end for g in clusters[ci])
            gap_end = min(g.start for g in clusters[ci + 1])
            if any(g.start < gap_start and g.end > gap_end for g in td_sorted):
                return [gene]

        # CRITICAL: Check for well-covered exons in each gap
        all_exons = []
        for tx in gene.transcripts:
            for e in tx.exons:
                all_exons.append((e.start, e.end))

        for ci in range(len(clusters) - 1):
            gap_start = max(g.end for g in clusters[ci]) + 1
            gap_end = min(g.start for g in clusters[ci + 1]) - 1
            if gap_end <= gap_start:
                continue
            for es, ee in all_exons:
                if es >= gap_start and ee <= gap_end and ee - es >= 5:
                    cov = self.coverage.get_mean_coverage(gene.seqid, es, ee)
                    if cov >= 10.0:
                        logger.debug(f"Split vetoed for {gene.gene_id}: "
                                   f"covered exon {es}-{ee} (cov={cov:.0f}) in gap")
                        return [gene]

        # ORF quality check: if the unsplit Helixer gene produces a single protein
        # substantially longer than any individual split fragment, keep unsplit.
        # This handles cases where StringTie splits a real gene due to coverage gaps.
        orf_finder = ORFFinder(self.genome)

        # Use the Helixer gene's exons directly for the best unsplit ORF
        # (consensus transcripts may not preserve the full Helixer exon set)
        helixer_id = gene.attributes.get('helixer_id', '')
        helixer_exons = None
        for hg in self.helixer_genes:
            if hg.gene_id == helixer_id and hg.transcripts:
                helixer_exons = hg.transcripts[0].exons
                break

        # Also check the consensus transcript with most exons
        best_tx = max(gene.transcripts, key=lambda tx: len(tx.exons))
        candidates = [best_tx.exons]
        if helixer_exons:
            candidates.append(helixer_exons)

        ev_starts = self.evidence_index.get_evidence_cds_starts(
            gene.seqid, gene.strand, gene.start, gene.end)
        unsplit_orf_len = 0
        for exon_set in candidates:
            orf = orf_finder.find_best_orf(gene.seqid, exon_set, gene.strand,
                                           evidence_cds_starts=ev_starts)
            if orf:
                unsplit_orf_len = max(unsplit_orf_len, orf[1] - orf[0])

        # Compute best ORF for each split cluster individually
        split_max_orf = 0
        for ci, cluster in enumerate(clusters):
            cl_start = min(g.start for g in cluster)
            cl_end = max(g.end for g in cluster)
            cl_ev_starts = self.evidence_index.get_evidence_cds_starts(
                gene.seqid, gene.strand, cl_start, cl_end)
            # Try both consensus exons and Helixer exons for each cluster
            for exon_set in candidates:
                cluster_exons = [e for e in exon_set
                                if e.start >= cl_start - 100 and e.end <= cl_end + 100]
                if cluster_exons:
                    cluster_orf = orf_finder.find_best_orf(gene.seqid, cluster_exons, gene.strand,
                                                           evidence_cds_starts=cl_ev_starts)
                    if cluster_orf:
                        split_max_orf = max(split_max_orf, cluster_orf[1] - cluster_orf[0])

        # Veto split if the unsplit gene produces a single protein longer than
        # the best individual fragment by at least 1.3x
        if (unsplit_orf_len > split_max_orf * 1.3
                and unsplit_orf_len >= 300
                and split_max_orf > 0):
            logger.info(f"Split vetoed for {gene.gene_id}: unsplit ORF "
                       f"{unsplit_orf_len}bp vs best fragment {split_max_orf}bp "
                       f"(ratio {unsplit_orf_len/split_max_orf:.1f}x)")
            return [gene]

        # All checks passed — split
        split_results = []
        for ci, cluster in enumerate(clusters):
            cl_start = min(g.start for g in cluster)
            cl_end = max(g.end for g in cluster)

            new_gene = Gene(
                gene_id=f"{gene.gene_id}_split{ci + 1}",
                seqid=gene.seqid, strand=gene.strand,
                start=cl_start, end=cl_end, source='Refined',
                attributes=dict(gene.attributes)
            )
            new_gene.attributes['ID'] = new_gene.gene_id

            for tx in gene.transcripts:
                cluster_exons = [
                    e for e in tx.exons
                    if e.start >= cl_start - 100 and e.end <= cl_end + 100
                ]
                if not cluster_exons:
                    continue
                new_tx = Transcript(
                    transcript_id=f"{new_gene.gene_id}.1",
                    seqid=gene.seqid, strand=gene.strand,
                    start=min(e.start for e in cluster_exons),
                    end=max(e.end for e in cluster_exons),
                    source='Refined'
                )
                new_tx.exons = cluster_exons
                new_tx.cds = [c for c in tx.cds
                             if c.start >= cl_start - 100 and c.end <= cl_end + 100]
                new_gene.transcripts.append(new_tx)

            if new_gene.transcripts:
                new_gene.start = min(e.start for tx in new_gene.transcripts for e in tx.exons)
                new_gene.end = max(e.end for tx in new_gene.transcripts for e in tx.exons)
                split_results.append(new_gene)
                logger.info(f"Split {gene.gene_id} -> {new_gene.gene_id} "
                           f"({new_gene.start}-{new_gene.end})")

        return split_results if split_results else [gene]

    def _drop_weak_premature_stop_exons(self, genes: List[Gene],
                                          orf_finder: 'ORFFinder') -> None:
        """Drop a weakly-supported internal exon if removing it yields a
        substantially longer ORF that reaches the terminal exon.

        Common failure mode: an alternative-isoform internal exon was
        included in the consensus, frame-shifts the CDS into a premature
        stop, and the remaining 3+ exons all become 3' UTR.  When the
        skipping junction (prev exon donor → next exon acceptor) is
        better supported than the inclusion junctions, dropping the exon
        recovers the dominant isoform.

        Triggered when ≥3 UTR exons exist on either end of the transcript.
        """
        UTR_EXON_TRIGGER = 3
        ORF_GAIN_FACTOR = 1.3       # new ORF must be ≥30% longer
        SKIP_RATIO = 1.3            # skip-junction reads must be ≥1.3× max(inclusion reads)
        SKIP_MIN_READS = 3          # absolute floor for skip support
        TERMINAL_REACH_FRAC = 0.85  # new CDS must extend to within last 15% of transcript

        n_dropped = 0
        for gene in genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            for tx in gene.transcripts:
                if not tx.cds or len(tx.exons) < 4:
                    continue

                sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                cds_min = min(c.start for c in tx.cds)
                cds_max = max(c.end for c in tx.cds)

                # Count UTR exons on each end (in transcript orientation)
                if gene.strand == '+':
                    n_5utr = sum(1 for e in sorted_exons if e.end < cds_min)
                    n_3utr = sum(1 for e in sorted_exons if e.start > cds_max)
                else:
                    n_5utr = sum(1 for e in sorted_exons if e.start > cds_max)
                    n_3utr = sum(1 for e in sorted_exons if e.end < cds_min)
                if max(n_5utr, n_3utr) < UTR_EXON_TRIGGER:
                    continue

                current_orf_len = sum(c.end - c.start + 1 for c in tx.cds)
                tx_total_len = sum(e.end - e.start + 1 for e in sorted_exons)

                ev_starts = self.evidence_index.get_evidence_cds_starts(
                    gene.seqid, gene.strand, gene.start, gene.end)

                best_drop_idx = None
                best_drop_orf_len = current_orf_len

                # Try dropping each internal exon (not first or last)
                for i in range(1, len(sorted_exons) - 1):
                    cand = sorted_exons[i]
                    prev_exon = sorted_exons[i - 1]
                    next_exon = sorted_exons[i + 1]

                    # Junction support assessment
                    if not self.bam_evidence.available:
                        # Without junction data we can still proceed using
                        # coverage as a fallback for "weak support".
                        cand_cov = self.coverage.get_mean_coverage(
                            gene.seqid, cand.start, cand.end)
                        ref_covs = sorted(
                            self.coverage.get_mean_coverage(gene.seqid, e.start, e.end)
                            for j, e in enumerate(sorted_exons) if j != i)
                        if not ref_covs:
                            continue
                        median_cov = ref_covs[len(ref_covs) // 2]
                        if cand_cov > 0.30 * median_cov:
                            continue
                        skip_supported = True  # coverage-only fallback
                    else:
                        incl_in = self.bam_evidence.reads_at_acceptor(
                            gene.seqid, cand.start, tolerance=2)
                        incl_out = self.bam_evidence.reads_at_donor(
                            gene.seqid, cand.end, tolerance=2)
                        max_incl = max(incl_in, incl_out)
                        # Look for the skip junction prev.end → next.start
                        skip_juncs = self.bam_evidence.find_junctions_starting_in(
                            gene.seqid,
                            prev_exon.end + 1 - 2,
                            prev_exon.end + 1 + 2,
                            min_reads=SKIP_MIN_READS)
                        skip_reads = 0
                        for js, je, jc in skip_juncs:
                            if abs(je - (next_exon.start - 1)) <= 2:
                                skip_reads = max(skip_reads, jc)
                        if skip_reads < SKIP_MIN_READS:
                            continue
                        if skip_reads < SKIP_RATIO * max(1, max_incl):
                            continue
                        skip_supported = True

                    # Tentatively build the exon-skipped transcript and
                    # find the best ORF.
                    test_exons = [e for j, e in enumerate(sorted_exons) if j != i]
                    test_orf = orf_finder.find_best_orf(
                        gene.seqid, test_exons, gene.strand,
                        coverage=self.coverage,
                        evidence_cds_starts=ev_starts)
                    if test_orf is None:
                        continue
                    new_cds_list = orf_finder.orf_to_genomic_cds(
                        gene.seqid, test_exons, gene.strand,
                        test_orf[0], test_orf[1])
                    if not new_cds_list:
                        continue
                    new_cds_len = sum(c.end - c.start + 1 for c in new_cds_list)

                    # Must be substantially longer
                    if new_cds_len < ORF_GAIN_FACTOR * current_orf_len:
                        continue

                    # New CDS must reach near the terminal exon (in
                    # transcript direction) — otherwise we haven't fixed
                    # the trailing-UTR problem.
                    new_cds_min = min(c.start for c in new_cds_list)
                    new_cds_max = max(c.end for c in new_cds_list)
                    test_sorted = sorted(test_exons, key=lambda e: e.start)
                    tx_start = test_sorted[0].start
                    tx_end = test_sorted[-1].end
                    if gene.strand == '+':
                        reach = (new_cds_max - tx_start) / max(1, tx_end - tx_start)
                    else:
                        reach = (tx_end - new_cds_min) / max(1, tx_end - tx_start)
                    if reach < TERMINAL_REACH_FRAC:
                        continue

                    if new_cds_len > best_drop_orf_len:
                        best_drop_idx = i
                        best_drop_orf_len = new_cds_len

                if best_drop_idx is not None:
                    dropped = sorted_exons[best_drop_idx]
                    logger.info(
                        f"  Step 5h.4: {gene.gene_id} dropping exon "
                        f"{dropped.start}-{dropped.end} "
                        f"(ORF {current_orf_len} → {best_drop_orf_len} nt)")
                    if self.tracer.enabled and self.tracer.matches(gene):
                        self.tracer.event(
                            "_drop_weak_premature_stop_exons",
                            f"DROPPED exon {dropped.start}-{dropped.end} from "
                            f"{gene.gene_id}; ORF {current_orf_len} → "
                            f"{best_drop_orf_len} nt")
                    tx.exons = [e for j, e in enumerate(sorted_exons)
                                if j != best_drop_idx]
                    tx.cds = []
                    tx.five_prime_utrs = []
                    tx.three_prime_utrs = []
                    n_dropped += 1

            # Re-derive CDS+UTRs if anything was dropped from this gene.
            if any(not tx.cds and tx.exons for tx in gene.transcripts):
                orf_finder.reassign_cds(gene, coverage=self.coverage,
                                         evidence_index=self.evidence_index)
                self._recompute_gene_boundaries(gene)

        if n_dropped:
            logger.info(f"  Step 5h.4: Dropped {n_dropped} weak internal exon(s)")

    def _add_alternative_isoforms(self, genes: List[Gene],
                                    orf_finder: 'ORFFinder') -> None:
        """Add alternative coding isoforms from TransDecoder evidence.

        For each refined gene, scan TD transcripts whose CDS substantially
        overlaps the primary CDS.  Add a TD transcript as an additional
        isoform when:
          - it differs from all already-accepted isoforms by ≥150 bp at
            either end, or has at least one unique internal splice site,
          - all of its introns have ≥3 junction reads (or it has no
            introns), and pass the stranded antisense veto if available,
          - its CDS yields an ORF ≥75% of the primary ORF length and ends
            in a stop codon.
        Caps at 3 isoforms per gene.
        """
        if not self.td_genes:
            return

        MIN_END_DIFF   = 150
        MAX_ISOFORMS   = 3
        MIN_REL_ORF    = 0.75
        MIN_JUNC_READS = 3
        MIN_CDS_OVL    = 0.50  # reciprocal overlap of CDS spans

        td_by_region = defaultdict(list)
        for tg in self.td_genes:
            td_by_region[(tg.seqid, tg.strand)].append(tg)

        n_added = 0
        for gene in genes:
            if gene.attributes.get('manual_annotation') == 'true':
                continue
            if not gene.transcripts or not gene.transcripts[0].cds:
                continue
            primary = gene.transcripts[0]
            primary_cds_min = min(c.start for c in primary.cds)
            primary_cds_max = max(c.end for c in primary.cds)
            primary_cds_len = sum(c.end - c.start + 1 for c in primary.cds)
            primary_span = primary_cds_max - primary_cds_min + 1

            candidates = []
            for tg in td_by_region.get((gene.seqid, gene.strand), []):
                if tg.end < gene.start or tg.start > gene.end:
                    continue
                for ttx in tg.transcripts:
                    if not ttx.cds or not ttx.exons:
                        continue
                    tcds_min = min(c.start for c in ttx.cds)
                    tcds_max = max(c.end for c in ttx.cds)
                    ov = min(tcds_max, primary_cds_max) - max(tcds_min, primary_cds_min)
                    if ov <= 0:
                        continue
                    td_span = tcds_max - tcds_min + 1
                    if ov < MIN_CDS_OVL * min(primary_span, td_span):
                        continue
                    candidates.append(ttx)

            if not candidates:
                continue

            # Sort candidates by ORF length, longest first
            candidates.sort(
                key=lambda t: -sum(c.end - c.start + 1 for c in t.cds))

            accepted = [primary]
            for cand in candidates:
                if len(accepted) >= MAX_ISOFORMS:
                    break
                if not all(self._isoforms_differ(cand, ex_tx, MIN_END_DIFF)
                           for ex_tx in accepted):
                    continue
                if not self._isoform_introns_supported(
                        gene.seqid, cand, MIN_JUNC_READS):
                    continue
                alt_orf_len = sum(c.end - c.start + 1 for c in cand.cds)
                if alt_orf_len < MIN_REL_ORF * primary_cds_len:
                    continue
                if not has_stop_codon(self.genome, gene.seqid, cand.cds, gene.strand):
                    continue
                # Reject alt isoforms whose own structure already has 3+
                # UTR exons on either end -- these are usually TD models
                # that themselves merge two genes (the trailing exons are a
                # separate gene's structure, not a real long UTR).
                cand_cds_min = min(c.start for c in cand.cds)
                cand_cds_max = max(c.end for c in cand.cds)
                cand_sorted = sorted(cand.exons, key=lambda e: e.start)
                if gene.strand == '+':
                    n5 = sum(1 for e in cand_sorted if e.end < cand_cds_min)
                    n3 = sum(1 for e in cand_sorted if e.start > cand_cds_max)
                else:
                    n5 = sum(1 for e in cand_sorted if e.start > cand_cds_max)
                    n3 = sum(1 for e in cand_sorted if e.end < cand_cds_min)
                if max(n5, n3) >= 3:
                    continue
                if not self._isoform_coverage_ok(gene, cand):
                    continue

                # Build a clean Transcript copy attached to this gene
                alt_exons = [Feature(seqid=e.seqid, source='Refined',
                                      ftype='exon',
                                      start=e.start, end=e.end,
                                      score=e.score, strand=e.strand,
                                      phase='.', attributes=dict(e.attributes))
                             for e in cand.exons]

                # TD transcripts often span only the CDS, missing the UTR
                # exons present in the primary refined transcript.  Inherit
                # primary's UTR exons that fall strictly outside the alt's
                # exonic range, and extend the alt's terminal exon outward
                # when it overlaps a primary exon that extends further into
                # the UTR (UTR within the same first/last exon).
                alt_lo = min(e.start for e in alt_exons)
                alt_hi = max(e.end for e in alt_exons)
                sorted_alt = sorted(alt_exons, key=lambda e: e.start)
                primary_sorted = sorted(primary.exons, key=lambda e: e.start)

                for pe in primary_sorted:
                    if pe.end < alt_lo:
                        alt_exons.append(Feature(
                            seqid=pe.seqid, source='Refined', ftype='exon',
                            start=pe.start, end=pe.end, score=pe.score,
                            strand=pe.strand, phase='.',
                            attributes=dict(pe.attributes)))
                    elif pe.start > alt_hi:
                        alt_exons.append(Feature(
                            seqid=pe.seqid, source='Refined', ftype='exon',
                            start=pe.start, end=pe.end, score=pe.score,
                            strand=pe.strand, phase='.',
                            attributes=dict(pe.attributes)))
                    else:
                        # Overlap with an alt exon -- extend the matching
                        # alt exon outward if primary extends further
                        if pe.start <= sorted_alt[0].end and pe.end >= sorted_alt[0].start:
                            if pe.start < sorted_alt[0].start:
                                sorted_alt[0].start = pe.start
                        if pe.start <= sorted_alt[-1].end and pe.end >= sorted_alt[-1].start:
                            if pe.end > sorted_alt[-1].end:
                                sorted_alt[-1].end = pe.end

                alt_exons.sort(key=lambda e: e.start)

                alt_tx = Transcript(
                    transcript_id=f"{gene.gene_id}.alt",
                    seqid=gene.seqid, strand=gene.strand,
                    start=min(e.start for e in alt_exons),
                    end=max(e.end for e in alt_exons),
                    source='Refined')
                alt_tx.exons = alt_exons
                alt_tx.cds = [Feature(seqid=c.seqid, source='Refined',
                                       ftype='CDS',
                                       start=c.start, end=c.end,
                                       score=c.score, strand=c.strand,
                                       phase=c.phase, attributes=dict(c.attributes))
                              for c in cand.cds]
                # Re-derive UTRs from expanded exons + CDS
                alt_tx.five_prime_utrs = []
                alt_tx.three_prime_utrs = []
                orf_finder._derive_utrs(alt_tx, gene.strand)
                # UTR inheritance can collapse the alt's distinguishing 5'/3'
                # boundary back onto the primary's boundary. Re-check that
                # the expanded alt is still meaningfully different.
                if not all(self._isoforms_differ(alt_tx, ex_tx, MIN_END_DIFF)
                           for ex_tx in accepted):
                    continue
                accepted.append(alt_tx)
                n_added += 1
                logger.info(
                    f"  Step 5j: {gene.gene_id} added alt isoform "
                    f"(exons={len(alt_tx.exons)}, ORF={alt_orf_len} nt, "
                    f"primary ORF={primary_cds_len} nt)")

            if len(accepted) > 1:
                gene.transcripts = accepted
                self._recompute_gene_boundaries(gene)

        if n_added:
            logger.info(f"  Step 5j: Added {n_added} alternative isoform(s)")

    @staticmethod
    def _isoforms_differ(tx_a, tx_b, min_end_diff: int) -> bool:
        """True if two transcripts differ enough to count as distinct
        isoforms: 5' or 3' boundary differs by ≥ min_end_diff bp, OR they
        have at least one unique internal splice site."""
        a_start = min(e.start for e in tx_a.exons)
        a_end   = max(e.end for e in tx_a.exons)
        b_start = min(e.start for e in tx_b.exons)
        b_end   = max(e.end for e in tx_b.exons)
        if abs(a_start - b_start) >= min_end_diff:
            return True
        if abs(a_end - b_end) >= min_end_diff:
            return True
        # Compare internal splice sites
        a_introns = set()
        sa = sorted(tx_a.exons, key=lambda e: e.start)
        for i in range(len(sa) - 1):
            a_introns.add((sa[i].end, sa[i+1].start))
        b_introns = set()
        sb = sorted(tx_b.exons, key=lambda e: e.start)
        for i in range(len(sb) - 1):
            b_introns.add((sb[i].end, sb[i+1].start))
        return bool(a_introns ^ b_introns)

    def _isoform_introns_supported(self, seqid: str, tx,
                                     min_reads: int) -> bool:
        """Every intron of `tx` has ≥ min_reads junction support."""
        if not self.bam_evidence.available:
            # No junctions data: can't validate, accept by default
            return True
        sorted_exons = sorted(tx.exons, key=lambda e: e.start)
        for i in range(len(sorted_exons) - 1):
            donor = sorted_exons[i].end
            acceptor = sorted_exons[i+1].start
            reads_d = self.bam_evidence.reads_at_donor(seqid, donor, tolerance=2)
            reads_a = self.bam_evidence.reads_at_acceptor(seqid, acceptor, tolerance=2)
            if min(reads_d, reads_a) < min_reads:
                return False
        return True

    def _isoform_coverage_ok(self, gene: Gene, tx) -> bool:
        """Mean coverage over alt-only exonic regions must clear the
        unstranded floor and pass the stranded veto if available."""
        if not self.coverage.available:
            return True
        for ex in tx.exons:
            cov = self.coverage.get_mean_coverage(gene.seqid, ex.start, ex.end)
            if cov < 1.0:
                return False
            if not self._passes_strand_check(
                    gene.seqid, ex.start, ex.end, gene.strand, cov):
                return False
        return True

    def _merge_alt_start_genes(self, genes: List[Gene]) -> List[Gene]:
        """Merge same-strand genes that share at least one exon exactly.

        Helixer (and occasionally other sources) sometimes splits a single
        biological gene into 2-3 adjacent predictions that share most of
        their downstream (or upstream) exons.  When two refined genes on
        the same strand share at least one exon with identical start/end
        coordinates, they are almost certainly alt-start/alt-end isoforms
        of one gene.  Cluster such genes via union-find on shared exon
        identity, pick the longest-ORF member as primary, and retain up to
        MAX_ISOFORMS additional members as alternative transcripts.
        """
        MAX_ISOFORMS = 3

        if len(genes) < 2:
            return genes

        # Index every transcript exon: (seqid, strand, start, end) -> set of gene idx
        exon_idx = defaultdict(set)
        for gi, g in enumerate(genes):
            if g.attributes.get('manual_annotation') == 'true':
                continue
            for tx in g.transcripts:
                for e in tx.exons:
                    exon_idx[(g.seqid, g.strand, e.start, e.end)].add(gi)

        # Union-find: gene index -> root
        parent = list(range(len(genes)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for shared in exon_idx.values():
            if len(shared) < 2:
                continue
            it = iter(shared)
            base = next(it)
            for other in it:
                union(base, other)

        # Group by root
        groups = defaultdict(list)
        for gi in range(len(genes)):
            groups[find(gi)].append(gi)

        merged = []
        consumed = set()
        n_merged = 0
        for root, members in groups.items():
            if len(members) < 2:
                continue
            member_genes = [genes[i] for i in members
                            if not consumed.intersection({i})]
            if not member_genes:
                continue
            # All in cluster must share strand+seqid (already enforced via key)
            # Pick primary as the one whose primary transcript has the longest ORF.
            def orf_len(g):
                if not g.transcripts or not g.transcripts[0].cds:
                    return 0
                return sum(c.end - c.start + 1 for c in g.transcripts[0].cds)
            member_genes.sort(key=lambda g: -orf_len(g))
            primary = member_genes[0]
            extras = member_genes[1:]

            primary_cds_min = (min(c.start for c in primary.transcripts[0].cds)
                                if primary.transcripts and primary.transcripts[0].cds
                                else None)
            primary_cds_max = (max(c.end for c in primary.transcripts[0].cds)
                                if primary.transcripts and primary.transcripts[0].cds
                                else None)

            for other in extras:
                if len(primary.transcripts) >= MAX_ISOFORMS:
                    break
                # Add each of `other`'s transcripts whose CDS substantially
                # overlaps primary's CDS span (>=50% reciprocal). Otherwise
                # the merge would conflate separate genes.
                added_any = False
                for otx in other.transcripts:
                    if not otx.exons:
                        continue
                    if otx.cds and primary_cds_min is not None:
                        ocds_min = min(c.start for c in otx.cds)
                        ocds_max = max(c.end for c in otx.cds)
                        ov = min(ocds_max, primary_cds_max) - max(ocds_min, primary_cds_min)
                        primary_span = primary_cds_max - primary_cds_min + 1
                        other_span = ocds_max - ocds_min + 1
                        if ov <= 0 or ov < 0.5 * min(primary_span, other_span):
                            continue
                    new_tx = Transcript(
                        transcript_id=f"{primary.gene_id}.alt",
                        seqid=primary.seqid, strand=primary.strand,
                        start=otx.start, end=otx.end, source='Refined')
                    new_tx.exons = list(otx.exons)
                    new_tx.cds = list(otx.cds)
                    new_tx.five_prime_utrs = list(otx.five_prime_utrs)
                    new_tx.three_prime_utrs = list(otx.three_prime_utrs)
                    primary.transcripts.append(new_tx)
                    added_any = True
                    if len(primary.transcripts) >= MAX_ISOFORMS:
                        break

                if added_any:
                    n_merged += 1
                    consumed.add(members[member_genes.index(other)])
                    logger.info(
                        f"  Step 5k: merged {other.gene_id} into "
                        f"{primary.gene_id} as alt isoform")

            self._recompute_gene_boundaries(primary)
            merged.append(primary)
            consumed.add(members[member_genes.index(primary)])

        # Anything not in a merge group, or merge primaries already added,
        # are returned as-is. consumed covers genes we've folded in.
        result = [merged[i] for i in range(len(merged))]
        for gi, g in enumerate(genes):
            if gi in consumed:
                continue
            result.append(g)

        if n_merged:
            logger.info(f"  Step 5k: Merged {n_merged} gene(s) into alt-start "
                        f"isoforms")
        return result

    def _split_excessive_utr_genes(self, genes: List[Gene]) -> List[Gene]:
        """Split genes flagged by an excessive UTR-exon count using StringTie evidence.

        A real gene almost never carries more than 2-3 exons of UTR on either
        end.  When the consensus shows 3+ UTR exons on one side of the CDS,
        the locus is typically two neighboring genes merged by a bridging
        template.  StringTie usually keeps them separate because it follows
        read coverage without chaining through cross-gene junctions.

        For each suspect gene: locate a pair of non-overlapping StringTie
        transcripts that together cover the locus, pick the inter-cluster
        gap that falls outside any CDS, partition the gene's exons at
        StringTie-informed boundaries, and re-validate each half (canonical
        splices, impossible introns, zero-coverage trim) before re-deriving
        CDS and UTRs via ORFFinder.  Veto the split if either half lacks a
        >=100 aa ORF or loses all exons during validation.
        """
        MIN_UTR_EXONS = 3
        MIN_ST_GAP = 50
        MIN_ORF_NT = 300

        result = []
        orf_finder = ORFFinder(self.genome)
        n_split = 0

        for gene in genes:
            if gene.attributes.get('manual_annotation') == 'true':
                result.append(gene)
                continue
            if not gene.transcripts:
                result.append(gene)
                continue

            trigger = any(
                len(tx.five_prime_utrs) >= MIN_UTR_EXONS
                or len(tx.three_prime_utrs) >= MIN_UTR_EXONS
                for tx in gene.transcripts)
            if not trigger:
                result.append(gene)
                continue

            # Collect StringTie transcripts inside gene footprint, same strand
            st_txs = []
            for stg in self.st_genes:
                if stg.seqid != gene.seqid:
                    continue
                if stg.strand != gene.strand and stg.strand != '.':
                    continue
                for stx in stg.transcripts:
                    if (stx.start >= gene.start - 100
                            and stx.end <= gene.end + 100
                            and stx.exons):
                        st_txs.append(stx)
            if len(st_txs) < 2:
                result.append(gene)
                continue

            # Cluster by transcript-level gaps
            st_txs.sort(key=lambda t: t.start)
            clusters = [[st_txs[0]]]
            for stx in st_txs[1:]:
                prev_end = max(t.end for t in clusters[-1])
                if stx.start > prev_end + MIN_ST_GAP:
                    clusters.append([stx])
                else:
                    clusters[-1].append(stx)
            if len(clusters) < 2:
                result.append(gene)
                continue

            did_split = False
            for ci in range(len(clusters) - 1):
                gap_start = max(t.end for t in clusters[ci]) + 1
                gap_end = min(t.start for t in clusters[ci + 1]) - 1
                if gap_end < gap_start:
                    continue

                cds_in_gap = any(
                    c.start <= gap_end and c.end >= gap_start
                    for tx in gene.transcripts for c in tx.cds)
                if cds_in_gap:
                    continue

                halves = self._build_utr_split_halves(
                    gene, gap_start, gap_end,
                    clusters[ci], clusters[ci + 1],
                    orf_finder, MIN_ORF_NT)
                if halves is None:
                    continue

                left_gene, right_gene = halves
                logger.info(
                    f"Step 5h.5: Splitting {gene.gene_id} at gap "
                    f"{gap_start}-{gap_end} -> "
                    f"{left_gene.gene_id} ({left_gene.start}-{left_gene.end}) + "
                    f"{right_gene.gene_id} ({right_gene.start}-{right_gene.end})")
                if self.tracer.enabled and self.tracer.matches(gene):
                    self.tracer.event(
                        "_split_excessive_utr_genes",
                        f"{gene.gene_id} -> {left_gene.gene_id} + {right_gene.gene_id} "
                        f"(gap {gap_start}-{gap_end})")
                result.extend([left_gene, right_gene])
                n_split += 1
                did_split = True
                break

            # If StringTie didn't yield a split, try TransDecoder evidence.
            # When a TD transcript has a coding ORF inside the gene's
            # footprint that doesn't overlap any current transcript's CDS,
            # that's strong evidence for a missed gene split.  The gap is
            # between the current gene CDS and the TD CDS.
            if not did_split:
                gene_cds_min = min(c.start for tx in gene.transcripts for c in tx.cds) \
                    if any(tx.cds for tx in gene.transcripts) else None
                gene_cds_max = max(c.end for tx in gene.transcripts for c in tx.cds) \
                    if any(tx.cds for tx in gene.transcripts) else None

                if gene_cds_min is not None:
                    td_alt_txs = []
                    for tg in self.td_genes:
                        if tg.seqid != gene.seqid:
                            continue
                        if tg.strand != gene.strand and tg.strand != '.':
                            continue
                        for ttx in tg.transcripts:
                            if not ttx.cds or not ttx.exons:
                                continue
                            tcds_min = min(c.start for c in ttx.cds)
                            tcds_max = max(c.end for c in ttx.cds)
                            # TD CDS must lie inside the gene footprint and
                            # be entirely outside the gene's CDS span
                            if tcds_min < gene.start or tcds_max > gene.end:
                                continue
                            if tcds_max >= gene_cds_min and tcds_min <= gene_cds_max:
                                continue
                            td_alt_txs.append((tcds_min, tcds_max, ttx))

                    for tcds_min, tcds_max, ttx in td_alt_txs:
                        # Determine left/right by genomic position
                        if tcds_max < gene_cds_min:
                            # TD is upstream of primary CDS
                            gap_start = tcds_max + 1
                            gap_end = gene_cds_min - 1
                            left_evidence = [ttx]
                            right_evidence = gene.transcripts
                        else:
                            # TD is downstream of primary CDS
                            gap_start = gene_cds_max + 1
                            gap_end = tcds_min - 1
                            left_evidence = gene.transcripts
                            right_evidence = [ttx]

                        if gap_end < gap_start:
                            continue

                        halves = self._build_utr_split_halves(
                            gene, gap_start, gap_end,
                            left_evidence, right_evidence,
                            orf_finder, MIN_ORF_NT)
                        if halves is None:
                            continue

                        left_gene, right_gene = halves
                        logger.info(
                            f"Step 5h.5 (TD): Splitting {gene.gene_id} at gap "
                            f"{gap_start}-{gap_end} -> "
                            f"{left_gene.gene_id} ({left_gene.start}-{left_gene.end}) + "
                            f"{right_gene.gene_id} ({right_gene.start}-{right_gene.end})")
                        if self.tracer.enabled and self.tracer.matches(gene):
                            self.tracer.event(
                                "_split_excessive_utr_genes (TD)",
                                f"{gene.gene_id} -> {left_gene.gene_id} + "
                                f"{right_gene.gene_id} (gap {gap_start}-{gap_end})")
                        result.extend([left_gene, right_gene])
                        n_split += 1
                        did_split = True
                        break

            if not did_split:
                result.append(gene)

        if n_split:
            logger.info(f"  Step 5h.5: Split {n_split} merged gene(s) "
                        f"via StringTie/TransDecoder evidence")
        return result

    def _build_utr_split_halves(self, gene: Gene, gap_start: int, gap_end: int,
                                 left_st: list, right_st: list,
                                 orf_finder: 'ORFFinder', min_orf_nt: int):
        """Build two validated half-genes from a gene partitioned at a gap.

        Uses StringTie exon boundaries to decide where exons that span the gap
        should be truncated.  Each half is re-validated with the same splice-
        canonical/coverage/intron helpers used in Step 2 and then handed to
        ORFFinder.reassign_cds to rebuild CDS + UTRs.  Returns None if either
        half fails ORF length, canonicalization, or coverage checks.
        """
        left_st_exons = [e for tx in left_st for e in tx.exons
                         if e.end <= gap_end]
        right_st_exons = [e for tx in right_st for e in tx.exons
                          if e.start >= gap_start]
        if not left_st_exons or not right_st_exons:
            return None
        left_boundary = max(e.end for e in left_st_exons)
        right_boundary = min(e.start for e in right_st_exons)
        if left_boundary >= right_boundary:
            return None

        base_tx = max(gene.transcripts, key=lambda t: len(t.exons))
        gene_exons = sorted(base_tx.exons, key=lambda e: e.start)

        left_exons, right_exons = [], []
        for e in gene_exons:
            if e.end <= left_boundary:
                left_exons.append(e)
            elif e.start >= right_boundary:
                right_exons.append(e)
            elif e.start < left_boundary and e.end > right_boundary:
                left_exons.append(Feature(
                    seqid=e.seqid, source=e.source, ftype='exon',
                    start=e.start, end=left_boundary, score=e.score,
                    strand=e.strand, phase=e.phase,
                    attributes=dict(e.attributes)))
                right_exons.append(Feature(
                    seqid=e.seqid, source=e.source, ftype='exon',
                    start=right_boundary, end=e.end, score=e.score,
                    strand=e.strand, phase=e.phase,
                    attributes=dict(e.attributes)))
            # else: exon entirely inside gap — drop
        if not left_exons or not right_exons:
            return None

        # ORF quality gate before we commit to the split
        for exons in (left_exons, right_exons):
            ev_starts = self.evidence_index.get_evidence_cds_starts(
                gene.seqid, gene.strand,
                min(e.start for e in exons), max(e.end for e in exons))
            orf = orf_finder.find_best_orf(
                gene.seqid, exons, gene.strand,
                coverage=self.coverage, evidence_cds_starts=ev_starts)
            if not orf or (orf[1] - orf[0] + 1) < min_orf_nt:
                return None

        def _finalize(suffix: str, exons: List[Feature]) -> Gene:
            new_gene = Gene(
                gene_id=f"{gene.gene_id}_split{suffix}",
                seqid=gene.seqid, strand=gene.strand,
                start=min(e.start for e in exons),
                end=max(e.end for e in exons),
                source='Refined',
                attributes=dict(gene.attributes))
            new_gene.attributes['ID'] = new_gene.gene_id
            new_gene.attributes['split_from'] = gene.gene_id
            new_tx = Transcript(
                transcript_id=f"{new_gene.gene_id}.1",
                seqid=gene.seqid, strand=gene.strand,
                start=new_gene.start, end=new_gene.end, source='Refined')
            new_tx.exons = list(exons)
            new_gene.transcripts.append(new_tx)

            new_tx.exons = [e for e in new_tx.exons if e.length >= MIN_EXON_SIZE]
            new_tx.exons = filter_impossible_introns(new_tx.exons)
            if self.coverage.available:
                new_tx.exons = self._merge_exons_by_coverage(
                    gene.seqid, new_tx.exons, gene.strand)
                new_tx = remove_zero_coverage_internal_exons(
                    new_tx, self.coverage, self.genome,
                    gene.seqid, gene.strand)
            new_tx.exons = enforce_canonical_splice_sites(
                self.genome, gene.seqid, new_tx.exons, gene.strand,
                bam_evidence=self.bam_evidence)
            new_tx.exons = self._remove_noncanonical_exons(
                gene.seqid, new_tx.exons, gene.strand)
            new_tx = trim_zero_coverage_terminal_exons(
                new_tx, self.coverage, gene.seqid, gene.strand)

            new_gene = orf_finder.reassign_cds(
                new_gene, coverage=self.coverage,
                evidence_index=self.evidence_index)
            self._recompute_gene_boundaries(new_gene)
            return new_gene

        left_gene = _finalize('A', left_exons)
        right_gene = _finalize('B', right_exons)

        for g in (left_gene, right_gene):
            if (not g.transcripts or not g.transcripts[0].exons
                    or not g.transcripts[0].cds):
                return None
            cds_len = sum(c.length for c in g.transcripts[0].cds)
            if cds_len < min_orf_nt:
                return None

        return left_gene, right_gene

    def _evaluate_merges(self, genes: List[Gene]) -> List[Gene]:
        """Evaluate and perform gene merges."""
        # Sort genes by position
        genes.sort(key=lambda g: (g.seqid, g.strand, g.start))

        merged_set = set()
        result = []
        i = 0

        while i < len(genes):
            if i in merged_set:
                i += 1
                continue

            current = genes[i]
            merge_count = 0  # Track chain merge depth

            # Look at next few genes for potential merges
            for j in range(i + 1, min(i + 4, len(genes))):
                if j in merged_set:
                    continue

                candidate = genes[j]
                if candidate.seqid != current.seqid:
                    break
                if candidate.strand != current.strand:
                    continue

                # Limit chain merges to 2 to avoid runaway merging
                if merge_count >= 2:
                    break

                # Skip over intervening single-exon genes (potential TEs)
                intervening = [
                    g for k, g in enumerate(genes[i + 1:j], i + 1)
                    if k not in merged_set and g.seqid == current.seqid
                ]
                skip_ok = all(
                    len(g.transcripts[0].exons) == 1
                    if g.transcripts and g.transcripts[0].exons
                    else True
                    for g in intervening
                )

                if not skip_ok:
                    continue

                if self.merger.should_merge(current, candidate):
                    logger.info(f"Merging {current.gene_id} + {candidate.gene_id}")
                    if self.tracer.enabled and self.tracer.pair_matches(current, candidate):
                        self.tracer.event(
                            "Step 5 merge",
                            f"MERGING {current.gene_id} + {candidate.gene_id} "
                            f"at gap {current.end + 1}-{candidate.start - 1}")
                    current = self.merger.merge_genes(current, candidate)
                    if self.tracer.matches(current):
                        self.tracer.snapshot("Step 5 post-merge", [current])
                    merged_set.add(j)
                    merge_count += 1

            result.append(current)
            i += 1

        return result

    def _score_ncrna(self, gene: Gene) -> float:
        """Score an ncRNA candidate."""
        if not gene.transcripts or not gene.transcripts[0].exons:
            return 0.0

        tx = gene.transcripts[0]
        exons = tx.sorted_exons()

        # Coverage-based scoring
        coverages = []
        for exon in exons:
            cov = self.coverage.get_mean_coverage(gene.seqid, exon.start, exon.end)
            coverages.append(cov)

        mean_cov = np.mean(coverages) if coverages else 0.0

        # Multi-exon ncRNAs are more believable
        exon_bonus = min(0.2, 0.1 * (len(exons) - 1))

        # Coverage score
        if mean_cov > 20.0:
            cov_score = 0.8
        elif mean_cov > 10.0:
            cov_score = 0.6
        elif mean_cov > 5.0:
            cov_score = 0.4
        else:
            cov_score = 0.2

        # Check splice site quality for multi-exon ncRNAs
        splice_score = 0.5
        if len(exons) > 1:
            introns = tx.introns()
            good_splices = 0
            for istart, iend in introns:
                seq_start = self.genome.get_sequence(gene.seqid, istart, istart + 1)
                seq_end = self.genome.get_sequence(gene.seqid, iend - 1, iend)
                if gene.strand == '-':
                    seq_start, seq_end = reverse_complement(seq_end), reverse_complement(seq_start)
                if seq_start.upper() == 'GT' and seq_end.upper() == 'AG':
                    good_splices += 1
            if introns:
                splice_score = 0.3 + 0.7 * (good_splices / len(introns))

        posterior = 0.4 * cov_score + 0.3 * splice_score + 0.3 * exon_bonus + 0.1
        return min(1.0, max(0.0, posterior))

    def _annotate_features(self, gene: Gene):
        """Add posterior probability annotations to all features of a gene."""
        eidx = self.evidence_index
        for tx in gene.transcripts:
            # Score each exon
            for exon in tx.exons:
                h_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='Helixer')
                t_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='TransDecoder')
                s_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, exon.start, exon.end, source='StringTie')
                exon.score = self.posterior_calc.score_exon(
                    gene.seqid, exon, gene.strand, h_support, t_support, s_support)

            # Score each CDS
            for cds in tx.cds:
                h_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, cds.start, cds.end, source='Helixer')
                t_support = eidx.has_overlapping_exon(
                    gene.seqid, gene.strand, cds.start, cds.end, source='TransDecoder')
                cds.score = self.posterior_calc.score_cds(
                    gene.seqid, cds, gene.strand, h_support, t_support)

            # Store intron scores in gene attributes
            for intron_start, intron_end in tx.introns():
                h_support = eidx.has_matching_intron(
                    gene.seqid, gene.strand, intron_start, intron_end, source='Helixer')
                s_support = eidx.has_matching_intron(
                    gene.seqid, gene.strand, intron_start, intron_end, source='StringTie')
                intron_score = self.posterior_calc.score_intron(
                    gene.seqid, intron_start, intron_end, gene.strand, h_support, s_support)
                gene.attributes[f'intron_{intron_start}_{intron_end}'] = f"{intron_score:.3f}"


# ============================================================================
# GFF3 output
# ============================================================================
def write_refined_gff(genes: List[Gene], output_path: str):
    """Write refined gene models to GFF3 format."""
    with open(output_path, 'w') as f:
        f.write("##gff-version 3\n")
        f.write(f"# Gene annotation refined by GeneAnnotationRefiner\n")
        f.write(f"# Posterior probability thresholds applied\n")
        f.write(f"# protein_coding genes and ncRNA genes included\n")

        for gene in genes:
            # Validate gene bounds before writing
            if (not isinstance(gene.start, int) or not isinstance(gene.end, int)
                    or gene.start > gene.end or gene.start < 1):
                logger.warning(f"  Skipping {gene.gene_id} in output: invalid bounds "
                             f"start={gene.start} end={gene.end}")
                continue

            # Gene line
            gene_attrs = f"ID={gene.gene_id}"
            gene_attrs += f";gene_biotype={gene.gene_type}"
            gene_attrs += f";posterior={gene.posterior:.4f}"
            if 'evidence_sources' in gene.attributes:
                gene_attrs += f";evidence_sources={gene.attributes['evidence_sources']}"
            if 'merged_from' in gene.attributes:
                gene_attrs += f";merged_from={gene.attributes['merged_from']}"
            if 'helixer_id' in gene.attributes:
                gene_attrs += f";helixer_id={gene.attributes['helixer_id']}"
            if 'transdecoder_id' in gene.attributes:
                gene_attrs += f";transdecoder_id={gene.attributes['transdecoder_id']}"
            if gene.gene_type == 'ncRNA':
                if 'mean_coverage' in gene.attributes:
                    gene_attrs += f";mean_coverage={gene.attributes['mean_coverage']}"
                if 'FPKM' in gene.attributes:
                    gene_attrs += f";FPKM={gene.attributes['FPKM']}"

            f.write(f"{gene.seqid}\tRefined\tgene\t{gene.start}\t{gene.end}\t"
                   f"{gene.posterior:.4f}\t{gene.strand}\t.\t{gene_attrs}\n")

            for tx in gene.transcripts:
                # mRNA/transcript line
                tx_type = 'mRNA' if gene.gene_type == 'protein_coding' else 'transcript'
                tx_attrs = f"ID={tx.transcript_id};Parent={gene.gene_id}"
                f.write(f"{gene.seqid}\tRefined\t{tx_type}\t{tx.start}\t{tx.end}\t"
                       f"{gene.posterior:.4f}\t{gene.strand}\t.\t{tx_attrs}\n")

                # Exons (sorted by position)
                for i, exon in enumerate(sorted(tx.exons, key=lambda e: e.start), 1):
                    post = exon.attributes.get('exon_posterior', '.')
                    exon_attrs = (f"ID={tx.transcript_id}.exon.{i};"
                                 f"Parent={tx.transcript_id};"
                                 f"exon_posterior={post}")
                    f.write(f"{gene.seqid}\tRefined\texon\t{exon.start}\t{exon.end}\t"
                           f"{exon.score:.4f}\t{gene.strand}\t.\t{exon_attrs}\n")

                # 5' UTRs
                for i, utr in enumerate(sorted(tx.five_prime_utrs, key=lambda u: u.start), 1):
                    utr_attrs = f"ID={tx.transcript_id}.five_prime_UTR.{i};Parent={tx.transcript_id}"
                    f.write(f"{gene.seqid}\tRefined\tfive_prime_UTR\t{utr.start}\t{utr.end}\t"
                           f".\t{gene.strand}\t.\t{utr_attrs}\n")

                # CDS features
                for i, cds in enumerate(sorted(tx.cds, key=lambda c: c.start), 1):
                    cds_attrs = f"ID={tx.transcript_id}.CDS.{i};Parent={tx.transcript_id}"
                    phase = cds.phase if cds.phase in ('0', '1', '2') else '.'
                    f.write(f"{gene.seqid}\tRefined\tCDS\t{cds.start}\t{cds.end}\t"
                           f"{cds.score:.4f}\t{gene.strand}\t{phase}\t{cds_attrs}\n")

                # 3' UTRs
                for i, utr in enumerate(sorted(tx.three_prime_utrs, key=lambda u: u.start), 1):
                    utr_attrs = f"ID={tx.transcript_id}.three_prime_UTR.{i};Parent={tx.transcript_id}"
                    f.write(f"{gene.seqid}\tRefined\tthree_prime_UTR\t{utr.start}\t{utr.end}\t"
                           f".\t{gene.strand}\t.\t{utr_attrs}\n")

    logger.info(f"Wrote refined GFF to {output_path}")


# ============================================================================
# Summary statistics
# ============================================================================
def print_summary(genes: List[Gene]):
    """Print summary statistics of refined annotation."""
    coding = [g for g in genes if g.gene_type == 'protein_coding']
    ncrna = [g for g in genes if g.gene_type == 'ncRNA']

    logger.info("\n" + "=" * 60)
    logger.info("ANNOTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total genes: {len(genes)}")
    logger.info(f"  Protein-coding: {len(coding)}")
    logger.info(f"  ncRNA: {len(ncrna)}")

    if coding:
        posteriors = [g.posterior for g in coding]
        logger.info(f"\nProtein-coding gene posteriors:")
        logger.info(f"  Mean: {np.mean(posteriors):.3f}")
        logger.info(f"  Median: {np.median(posteriors):.3f}")
        logger.info(f"  Min: {np.min(posteriors):.3f}")
        logger.info(f"  Max: {np.max(posteriors):.3f}")

        # Exon counts
        exon_counts = []
        for g in coding:
            if g.transcripts:
                exon_counts.append(max(len(tx.exons) for tx in g.transcripts) if g.transcripts else 0)
        if exon_counts:
            logger.info(f"\nExons per gene:")
            logger.info(f"  Mean: {np.mean(exon_counts):.1f}")
            logger.info(f"  Median: {np.median(exon_counts):.1f}")
            logger.info(f"  Range: {min(exon_counts)}-{max(exon_counts)}")

        # Gene lengths
        gene_lens = [g.end - g.start + 1 for g in coding]
        logger.info(f"\nGene lengths (bp):")
        logger.info(f"  Mean: {np.mean(gene_lens):.0f}")
        logger.info(f"  Median: {np.median(gene_lens):.0f}")

        # Evidence source breakdown
        evidence_counts = defaultdict(int)
        for g in coding:
            sources = g.attributes.get('evidence_sources', 'Unknown')
            for src in sources.split(','):
                evidence_counts[src.strip()] += 1
        logger.info(f"\nEvidence sources supporting coding genes:")
        for src, cnt in sorted(evidence_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {src}: {cnt}")

    if ncrna:
        posteriors = [g.posterior for g in ncrna]
        logger.info(f"\nncRNA gene posteriors:")
        logger.info(f"  Mean: {np.mean(posteriors):.3f}")
        logger.info(f"  Median: {np.median(posteriors):.3f}")



def check_dependencies():
    """Check that all required Python packages are installed."""
    missing = []
    optional_missing = []

    # Required dependencies
    deps = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pyfaidx': 'pyfaidx',
        'pyBigWig': 'pyBigWig',
    }
    for import_name, pip_name in deps.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    # Optional dependencies
    try:
        __import__('pysam')
    except ImportError:
        optional_missing.append('pysam')

    if missing or optional_missing:
        print("=" * 70)
        print("Gene Annotation Refiner - Dependency Check")
        print("=" * 70)

    if missing:
        print()
        print(f"ERROR: {len(missing)} required package(s) not found:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install all required dependencies with:")
        print()
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Or install everything at once:")
        print()
        print("  pip install numpy scipy pyfaidx pyBigWig pysam")
        print()

    if optional_missing:
        print()
        print(f"WARNING: {len(optional_missing)} optional package(s) not found:")
        for pkg in optional_missing:
            print(f"  - {pkg}")
        print()
        if 'pysam' in optional_missing:
            print("  pysam is required only when using --bam for splice junction")
            print("  evidence from RNA-seq read alignments. Without it, the pipeline")
            print("  runs using coverage, annotation, and splice-site evidence only.")
            print()
            print("  Install with:  pip install pysam")
            print()
            print("  Note: pysam requires htslib. On some systems you may also need:")
            print("    Ubuntu/Debian: sudo apt-get install libhts-dev")
            print("    macOS:         brew install htslib")
            print()

    if missing or optional_missing:
        # Also check for samtools (external tool)
        import shutil
        if not shutil.which('samtools'):
            print("NOTE: samtools is not on your PATH. It is used for automatic")
            print("  SAM-to-BAM conversion and BAM indexing. Without it, you must")
            print("  provide a sorted, indexed BAM file directly.")
            print()
            print("  Install with:")
            print("    Ubuntu/Debian: sudo apt-get install samtools")
            print("    macOS:         brew install samtools")
            print("    conda:         conda install -c bioconda samtools")
            print()

        print("=" * 70)

    if missing:
        sys.exit(1)


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Gene Annotation Refiner - Integrate multiple evidence sources '
                    'to produce refined gene models with posterior probabilities.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full consensus mode (traditional):
  %(prog)s --genome genome.fa --helixer helixer.gff --stringtie stringtie.gtf \\
           --transdecoder transdecoder.gff --bigwig rnaseq.bw --output refined.gff

  # With manual annotations:
  %(prog)s --genome genome.fa --helixer helixer.gff --stringtie stringtie.gtf \\
           --transdecoder transdecoder.gff --bigwig rnaseq.bw \\
           --manual_annotation curated.gff --output refined.gff

  # Refine an existing annotation with RNA-seq data:
  %(prog)s --genome genome.fa --refine_existing previous.gff \\
           --bigwig rnaseq.bw --bam rnaseq.bam --output refined.gff

  # Renumber genes with custom prefix and inherit names:
  %(prog)s --genome genome.fa --refine_existing previous.gff \\
           --bigwig rnaseq.bw --output refined.gff \\
           --renumber --gene_prefix OFASC --name_from official_v1.gff

All GFF inputs are optional, but at least one of --helixer, --stringtie,
--transdecoder, --manual_annotation, or --refine_existing must be provided.
        """
    )
    # Required
    parser.add_argument('--genome', required=True, help='Genome FASTA file')
    parser.add_argument('--output', required=True, help='Output refined GFF3 file')

    # GFF inputs (all optional, but at least one required)
    gff_group = parser.add_argument_group('GFF inputs (at least one required)')
    gff_group.add_argument('--helixer', default=None,
                           help='Helixer GFF3 predictions')
    gff_group.add_argument('--stringtie', default=None,
                           help='StringTie GTF transcripts')
    gff_group.add_argument('--transdecoder', default=None,
                           help='TransDecoder GFF3')
    gff_group.add_argument('--manual_annotation', default=None,
                           help='GFF3 of human-curated gene models. Exons get high '
                                'confidence; UTR ends remain low confidence since '
                                'these are often estimated approximately.')
    gff_group.add_argument('--refine_existing', default=None,
                           help='Use an existing annotation GFF3 as the base '
                                'instead of building consensus from '
                                'helixer/stringtie/transdecoder. By default the '
                                'existing annotation is trusted and only manual '
                                'annotations replace overlapping genes; pass '
                                '--refine_with_evidence to also run RNA-seq '
                                'evidence-based refinement steps.')
    gff_group.add_argument('--refine_with_evidence', action='store_true',
                           help='In --refine_existing mode, additionally run '
                                'evidence-based refinement (junction validation, '
                                'coverage trimming, canonical splice enforcement, '
                                'etc.). Requires --bigwig.')

    # Evidence data
    parser.add_argument('--bigwig', default=None, nargs='+',
                        help='RNA-seq coverage bigwig(s) (recommended). One or '
                             'more files; values are summed across files. Built '
                             'from all libraries (stranded + unstranded) — used '
                             'as the primary coverage signal throughout the '
                             'pipeline.')
    parser.add_argument('--bigwig_fwd', default=None, nargs='+',
                        help='Optional same-strand bigwig(s) for transcripts on '
                             'the + strand (e.g. deepTools '
                             '`bamCoverage --filterRNAstrand forward` from '
                             'stranded libraries only). One or more files; '
                             'values are summed. Used as a veto: if same-strand '
                             'coverage is near-zero where the unstranded bigwig '
                             'shows support, the support is rejected as '
                             'antisense leakage. Applied in Phase 4.5 '
                             '(terminal-exon UTR extension) and Step 5g.5 '
                             '(downstream-exon recovery). Requires --bigwig_rev.')
    parser.add_argument('--bigwig_rev', default=None, nargs='+',
                        help='Same-strand bigwig(s) for transcripts on the − '
                             'strand. One or more files; values are summed. '
                             'Pairs with --bigwig_fwd.')
    parser.add_argument('--bam', default=None, nargs='+',
                        help='RNA-seq BAM file(s) (optional, for splice junction '
                             'evidence). One or more files; junction read '
                             'counts are summed across files.')
    parser.add_argument('--junctions', default=None, nargs='+',
                        help='Pre-computed splice junction file(s) (Portcullis '
                             '.tab, STAR SJ.out.tab, or BED). Much faster than '
                             '--bam. One or more files; junctions are merged '
                             'and read counts summed across files. If both '
                             '--bam and --junctions are provided, --junctions '
                             'is used and --bam is ignored.')

    # Naming and renumbering options
    name_group = parser.add_argument_group('Gene naming')
    name_group.add_argument('--renumber', action='store_true',
                            help='Renumber all genes in position order')
    name_group.add_argument('--gene_prefix', default='GENE',
                            help='Prefix for gene names when renumbering '
                                 '(default: GENE)')
    name_group.add_argument('--name_from', default=None,
                            help='GFF3 file to inherit gene names from. Genes '
                                 'overlapping models in this file inherit their '
                                 'names; new non-overlapping genes get sequential '
                                 'names based on the numbering scheme in this file.')

    # Scoring configuration
    parser.add_argument('--config', default=None,
                        help='Scoring configuration INI file (see --dump_config)')
    parser.add_argument('--dump_config', action='store_true',
                        help='Write default configuration to default_config.ini and exit')
    parser.add_argument('--pwm_organism', default='drosophila',
                        choices=['drosophila', 'human', 'arabidopsis'],
                        help='Reference organism for splice-site PWMs when insufficient '
                             'training data are available from StringTie (default: drosophila). '
                             'Use "drosophila" for insects, "human" for vertebrates, '
                             '"arabidopsis" for plants.')
    parser.add_argument('--ncRNA_threshold', type=float, default=0.20,
                        help='Posterior threshold for ncRNA (default: 0.20)')
    parser.add_argument('--coding_threshold', type=float, default=0.20,
                        help='Posterior threshold for coding genes (default: 0.20). '
                             'The posterior score is written to every gene and exon '
                             'in the output GFF, so downstream filtering by score is '
                             'straightforward.  Lower values retain more candidate '
                             'models; raise to 0.50 for a higher-confidence set.')

    # Debug / tracing
    trace_group = parser.add_argument_group(
        'Debug tracing',
        'Follow one or more genes through every pipeline step.  Matching '
        'is by substring, so passing a short prefix like "005173" matches '
        '"Apis_helixer_scaffold_2_005173.1" and any refined gene that '
        'inherited it via the merge trail.  When the refined/renumbered '
        'ID is all that is known, use --trace_region to follow a locus.  '
        'All trace output is prefixed with "[TRACE]".')
    trace_group.add_argument(
        '--trace_gene', action='append', default=[], metavar='GENE_ID',
        help='Gene ID (or substring) to trace.  Repeat for multiple genes.')
    trace_group.add_argument(
        '--trace_region', action='append', default=[], metavar='SEQID:START-END',
        help='Genomic region to trace (e.g. scaffold_2:77600000-77680000). '
             'Any gene overlapping the region is traced.  Repeatable.')

    # Handle --dump_config before dep check (no deps needed)
    if '--dump_config' in sys.argv:
        cfg = ScoringConfig()
        out_path = 'default_config.ini'
        cfg.write_default_config(out_path)
        print(f"Default configuration written to {out_path}")
        return

    check_dependencies()

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Upfront file existence check — fail fast with clear messages
    # rather than deep into a multi-hour run.
    # ----------------------------------------------------------------
    file_args = {
        '--genome':             args.genome,
        '--helixer':            args.helixer,
        '--stringtie':          args.stringtie,
        '--transdecoder':       args.transdecoder,
        '--bigwig':             args.bigwig,
        '--bigwig_fwd':         args.bigwig_fwd,
        '--bigwig_rev':         args.bigwig_rev,
        '--bam':                args.bam,
        '--junctions':          args.junctions,
        '--manual_annotation':  args.manual_annotation,
        '--refine_existing':    args.refine_existing,
        '--name_from':          args.name_from,
        '--config':             args.config,
    }
    missing = []
    for flag, val in file_args.items():
        if not val:
            continue
        paths = [val] if isinstance(val, str) else list(val)
        for path in paths:
            if not os.path.exists(path):
                missing.append(f"  {flag}: {path}")
    if missing:
        logger.error("The following input files were not found — aborting before "
                     "starting the pipeline:\n" + "\n".join(missing))
        logger.error("Tip: inline bash comments after a backslash continuation "
                     r"(e.g.  --flag value \ #comment) silently break the "
                     "argument — move comments to their own line.")
        sys.exit(1)
    else:
        logger.info("All provided input files found.")

    # Validate: at least one GFF input must be provided
    gff_inputs = [args.helixer, args.stringtie, args.transdecoder,
                  args.manual_annotation, args.refine_existing]
    if not any(gff_inputs):
        parser.error("At least one GFF input is required: --helixer, --stringtie, "
                     "--transdecoder, --manual_annotation, or --refine_existing")

    # Validate: --refine_existing is incompatible with consensus sources
    if args.refine_existing and (args.helixer or args.stringtie or args.transdecoder):
        logger.warning("--refine_existing mode: ignoring --helixer, --stringtie, "
                      "and --transdecoder (using existing annotation as base)")

    # Validate: stranded bigwigs come as a pair
    if bool(args.bigwig_fwd) != bool(args.bigwig_rev):
        parser.error("--bigwig_fwd and --bigwig_rev must be provided together")

    # Validate: --refine_with_evidence requires --refine_existing
    if args.refine_with_evidence and not args.refine_existing:
        parser.error("--refine_with_evidence requires --refine_existing")
    if args.refine_with_evidence and not args.bigwig:
        parser.error("--refine_with_evidence requires --bigwig")

    # Validate: --name_from requires --renumber
    if args.name_from and not args.renumber:
        parser.error("--name_from requires --renumber")

    # Load scoring configuration
    if args.config:
        logger.info(f"Loading scoring config from: {args.config}")
        scoring_config = ScoringConfig.from_file(args.config)
    else:
        scoring_config = ScoringConfig()

    # CLI thresholds always override config file values
    scoring_config.ncrna_threshold = args.ncRNA_threshold
    scoring_config.coding_threshold = args.coding_threshold

    scoring_config.validate_weights()
    scoring_config.log_active_config()

    # Build the gene tracer from --trace_gene / --trace_region flags
    trace_regions = []
    for r in args.trace_region or []:
        try:
            trace_regions.append(GeneTracer.parse_region(r))
        except ValueError as e:
            parser.error(str(e))
    tracer = GeneTracer(gene_ids=args.trace_gene or [], regions=trace_regions)

    refiner = GeneAnnotationRefiner(
        genome_path=args.genome,
        helixer_path=args.helixer if not args.refine_existing else None,
        stringtie_path=args.stringtie if not args.refine_existing else None,
        transdecoder_path=args.transdecoder if not args.refine_existing else None,
        bigwig_path=args.bigwig,
        bigwig_fwd_path=args.bigwig_fwd,
        bigwig_rev_path=args.bigwig_rev,
        bam_path=args.bam,
        junctions_path=args.junctions,
        manual_annotation_path=args.manual_annotation,
        refine_existing_path=args.refine_existing,
        evidence_refinement=(not args.refine_existing) or args.refine_with_evidence,
        scoring_config=scoring_config,
        pwm_organism=args.pwm_organism,
        tracer=tracer,
    )

    refined_genes = refiner.refine()

    # Apply renumbering/naming if requested
    if args.renumber:
        name_from_genes = None
        if args.name_from:
            logger.info(f"Loading name reference from: {args.name_from}")
            name_from_genes = parse_generic_gff3(args.name_from,
                                                  source_label='NameRef')
            logger.info(f"  Loaded {len(name_from_genes)} reference gene names")
        refined_genes = renumber_genes(refined_genes, prefix=args.gene_prefix,
                                       name_from_genes=name_from_genes)
        tracer.snapshot("After renumber", refined_genes)

    write_refined_gff(refined_genes, args.output)
    print_summary(refined_genes)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
