# Gene Annotation Refiner

A Python pipeline that integrates multiple gene model outputs (gff files) and evidence sources (RNA seq converage and junctions) to produce a refined gene annotation. Each element receives a support score from 0-1.

## Overview

Annotation of new genomes is challenging and no single gene annotation pipeline provides results that are consistent with evidence. This pipeline combines gene model predictions from multiple pipelines, provided as gff files, and determines to what extent RNAseq evidence support each element. Genes are combined or split based on junction evidence and terminal exons are added (or removed) based on presence (or absence) of RNA seq evidence for additional exons beyond the prediction boundaries of any single gff file. Each gene, transcript, exon, intron, and CDS are provided with a confidence score (between 0-1) reflecting the strength of multi-source evidence.

**Supported evidence types**

| Input | Flag | Purpose |
|---|---|---|
| Genome FASTA | `--genome` | Genome sequence for splice-site PWM scoring and ORF finding |
| Helixer predictions | `--helixer` | ML-based gene structure (coding regions, intron/exon boundaries) |
| StringTie transcripts | `--stringtie` | RNA-seq-derived transcript models |
| TransDecoder CDS | `--transdecoder` | Exons and CDS predictions from long-read or short-read RNA-seq |
| RNA-seq BigWig | `--bigwig` | Per-base read coverage for exon validation |
| Portcullis/STAR junctions | `--junctions` | Pre-computed splice junction read counts (recommended) |
| BAM alignments | `--bam` | Alternative to `--junctions`; slower, re-scans reads per intron |
| Manual annotation | `--manual_annotation` | Human-curated models; propagated at high confidence |
| Existing GFF | `--refine_existing` | Refine a previous annotation rather than building from scratch |

---

## Installation

```bash
# Create environment
conda create -n refine_annot_env -c conda-forge -c bioconda \
    python=3.10 numpy scipy pyfaidx pybigwig pysam

conda activate refine_annot_env
```

`pysam` is only required when using `--bam`. All other functionality is available without it.

---

## Usage

### Mode 1 — Full consensus (build from scratch)

```bash
python gene_annotation_refiner_2.py \
    --genome genome.fa \
    --output refined_annotation.gff \
    --helixer helixer.gff \
    --stringtie stringtie.gtf \
    --transdecoder transdecoder.gff3 \
    --bigwig rnaseq.bw \
    --junctions portcullis_filtered.pass.junctions.tab \
    --renumber \
    --gene_prefix MySpecies
```

### Mode 2 — Refine an existing annotation

```bash
python gene_annotation_refiner_2.py \
    --genome genome.fa \
    --output refined_v2.gff \
    --refine_existing previous_annotation.gff \
    --bigwig rnaseq.bw \
    --junctions portcullis_filtered.pass.junctions.tab
```

### Splice junction input

Portcullis is the recommended source for junction evidence. Use the `.tab` output (not `.bed`): the tab file contains raw alignment counts (`nb_raw_aln` column), which the pipeline uses for evidence calibration. The `.bed` file contains only Portcullis confidence scores and is not useful here.

```bash
portcullis full --threads 8 genome.fa rnaseq.bam
# Use: portcullis_out/3-filt/portcullis_filtered.pass.junctions.tab
```

STAR `SJ.out.tab` files are also accepted.

---

## All options

```
Required:
  --genome FILE          Genome FASTA file
  --output FILE          Output refined GFF3 file

GFF inputs (at least one required):
  --helixer FILE         Helixer GFF predictions
  --stringtie FILE       StringTie GTF transcripts
  --transdecoder FILE    TransDecoder GFF3 CDS predictions
  --manual_annotation FILE  Manually curated GFF3 (propagated at high confidence)
  --refine_existing FILE Refine this existing annotation (Mode 2)

Evidence:
  --bigwig FILE          RNA-seq BigWig coverage file
  --junctions FILE       Pre-computed junction file (Portcullis .tab, STAR SJ.out.tab, or BED)
  --bam FILE             BAM file for splice junction counts (slower; use --junctions when possible)

Gene naming:
  --renumber             Renumber all output genes in genomic position order
  --gene_prefix PREFIX   Prefix for gene names (default: GENE)
  --name_from FILE       Inherit names from this GFF for overlapping genes; new genes get new names

Scoring:
  --config FILE          INI configuration file to override scoring parameters
  --dump_config          Write default config to gene_annotation_refiner_default.ini and exit
  --coding_threshold N   Minimum posterior for protein-coding genes (default: 0.20)
  --ncRNA_threshold N    Minimum posterior for ncRNA genes (default: 0.20)
```

---

## The refinement pipeline

The pipeline runs in 9 numbered steps after an initial gene-building phase. Here is what each step does and why.

### Initialization

**Calibration.** Before building any models, the pipeline fits empirical CDFs over two distributions derived from all StringTie templates:
- *Template posterior* — the 5th percentile sets `drop_threshold`: the minimum per-exon posterior below which a candidate exon is discarded.
- *Mean junction support* — the 1st percentile (floor: 1.0) sets `template_min_junction_mean`: multi-exon templates whose average junction read count falls below this threshold are rejected as low-quality.

**Splice-site PWMs.** Donor and acceptor position weight matrices are trained from all GT-AG (and GC-AG) dinucleotides flanking confirmed introns in the input data. These are used throughout to score candidate splice sites.

**Evidence index.** All evidence is loaded into an interval-based spatial index (binary-search over sorted start positions), replacing O(n) linear scans with O(log n + k) lookups for every overlap query.

---

### `_build_consensus` — constructing gene models

This internal method is the core of the pipeline and itself runs in 4 phases:

**Phase 1: Collect candidate exons**

All exons from all evidence sources are merged into a non-redundant set. Near-duplicate exons (within `DEDUP_TOL = 50 bp`) are collapsed into a single candidate; when two sources disagree on an exon boundary, the better boundary is chosen by a composite score:

- *Coverage drop*: fraction by which RNA-seq coverage falls on the intron side relative to the exon body. A sharp drop indicates a real boundary.
- *Splice-site PWM*: donor/acceptor log-likelihood from strand-aware PWMs trained on the input data.

The composite score is `0.6 × coverage_drop + 0.4 × PWM_norm`.

**Phase 2: Score candidate exons**

Each candidate exon receives a posterior from a weighted sum of four evidence likelihoods:

| Component | Weight | Score |
|---|---|---|
| Helixer overlap | 0.30 | 0.85 (supported) / 0.15 (not) |
| TransDecoder overlap | 0.25 | 0.80 / 0.20 |
| StringTie overlap | 0.25 | 0.80 / 0.20 |
| BigWig coverage | 0.20 | 0.90 / 0.70 / 0.50 / 0.20 (4 tiers) |

Exons falling below `drop_threshold` are discarded.

**Phase 3: Assemble gene models from templates**

Each multi-exon StringTie or Helixer transcript is used as a *template*. The pipeline walks the template's exons and for each one finds the highest-scoring candidate exon in Phase 2's pool that overlaps it. The assembled set of best-matching candidates forms the refined exon chain for that template.

Low-quality multi-exon templates (mean junction support < `template_min_junction_mean`) are skipped.

**Terminal exon junction audit.** Before any CDS logic, the assembled exon chain is trimmed from both ends: any terminal exon whose connecting intron has zero junction reads is stripped. This prevents spurious terminal exons assembled without RNA-seq support from entering the model.

**CDS assignment.** The template's CDS coordinates are projected onto the refined exon set. If the template CDS starts at exon index ≥ 3 in the refined set (indicating the exon chain changed significantly relative to the template), the CDS is recalculated from scratch by finding the earliest ATG that yields a valid ORF (≥ 150 bp) using `find_best_orf`.

`find_best_orf` scans every position in the spliced transcript sequence for an ATG, collects all ORFs ≥ 150 bp with an in-frame stop codon (or running to the end of the transcript), and returns the one with the earliest start position (longest ORF as tiebreaker). This ensures the translation start site is not biased toward a spuriously long internal ORF.

**Phase 4: Resolve overlapping models**

When multiple templates produce overlapping refined models, the best-scoring model is kept. Scoring penalizes models with terminal exons lacking junction support.

---

### Step 1: Build consensus gene models

Runs `_build_consensus` on all evidence genes. In Mode 2 (`--refine_existing`), the existing annotation is loaded first and used as an additional evidence source.

**Step 1b** optionally splits genes where StringTie or TransDecoder place separate, non-overlapping models — a heuristic for detecting Helixer-fused gene pairs.

---

### Step 2: Filter impossible introns and enforce canonical splice sites

Introns shorter than a minimum length or with non-canonical splice sites (not GT-AG or GC-AG) are removed. Where possible the pipeline attempts to find a nearby GT-AG dinucleotide within ±10 bp and adjusts the exon boundary rather than discarding the intron outright.

---

### Step 3: Evaluate splice sites and UTR boundaries

Each intron is scored for:
- Splice-site motif (GT-AG = 0.90, GC-AG = 0.70, other = 0.20)
- Splice-site PWM log-likelihood
- BigWig coverage continuity across the intron
- Junction read count from `--junctions` or `--bam`

Introns that fail all evidence criteria are removed and the flanking exons merged.

---

### Step 4: Recover UTRs from StringTie

StringTie often assembles UTR sequence that Helixer does not predict. This step extends gene models at their 5′ and 3′ ends by appending StringTie exons that lie beyond the current CDS boundary.

**Junction guard.** Before appending any UTR exon, the pipeline checks that the intron connecting it to the next exon has at least one junction read. This prevents re-introduction of the same spurious terminal exons removed in the Phase 3 terminal audit.

---

### Step 5: Gene merging, filtering, and cleanup

A sequence of sub-steps that refine the gene set:

| Sub-step | What it does |
|---|---|
| **5b** | Re-validate splice sites after any merges; remove exons that now violate splice constraints |
| **5c** | Merge adjacent exons where BigWig coverage is continuous and no splice site separates them |
| **5d** | Remove exons with no RNA-seq coverage *and* no StringTie/TransDecoder match |
| **5e** | Split genes where isoforms no longer overlap genomically |
| **5f** | Rank and select the single best isoform per gene |
| **5g** | Re-derive CDS from the best ORF in the final refined transcript |
| **5h** | Split genes where retained isoforms have completely non-overlapping CDS |
| **5i** | Repair transcripts that lost exons; recompute gene boundaries |

---

### Step 6: Compute posterior probabilities

Each gene receives a posterior combining exon quality, intron quality, internal consistency, and (if available) junction evidence. Single-exon genes use a reduced two-component model (exon quality + consistency). The gene posterior is the primary metric used for filtering and is written to the `posterior=` attribute in the output GFF.

---

### Step 7: Filter by posterior threshold

Genes below `--coding_threshold` (default 0.20) are discarded. Because posterior scores are written to the GFF, users can post-filter at any threshold without re-running the pipeline:

```bash
awk -F'\t' '$3=="gene" && /posterior=/' refined.gff | \
    awk -F'posterior=' '{split($2,a,";"); if(a[1]+0 >= 0.50) print}' > high_confidence.gff
```

---

### Step 8: Detect ncRNA candidates

Genomic regions with RNA-seq coverage (mean ≥ 3×) that are not explained by any protein-coding model are flagged as putative ncRNAs if they also meet an FPKM threshold. These are written with `gene_biotype=ncRNA` and their own posterior score.

---

### Step 9: Compute per-feature posteriors

Individual exon, intron, and CDS features each receive their own `exon_posterior=` / `intron_posterior=` / `cds_posterior=` attribute, allowing fine-grained downstream filtering by feature rather than by gene.

---

## Output GFF3 format

The output is standard GFF3 with additional attributes on each feature:

```
scaffold_2  Refined  gene   1234567  1245678  0.73  +  .  ID=MySpecies_000001;gene_biotype=protein_coding;posterior=0.73;evidence_sources=Helixer,StringTie,TransDecoder
scaffold_2  Refined  mRNA   1234567  1245678  0.73  +  .  ID=MySpecies_000001.1;Parent=MySpecies_000001
scaffold_2  Refined  exon   1234567  1234823  0.81  +  .  ID=MySpecies_000001.1.exon.1;Parent=MySpecies_000001.1;exon_posterior=0.81
scaffold_2  Refined  CDS    1234567  1234823  0.88  +  0  ID=MySpecies_000001.1.CDS.1;Parent=MySpecies_000001.1;cds_posterior=0.88
```

Key attributes:

| Attribute | Feature | Meaning |
|---|---|---|
| `posterior` | gene | Gene-level posterior probability (0–1) |
| `gene_biotype` | gene | `protein_coding` or `ncRNA` |
| `evidence_sources` | gene | Which sources contributed (e.g. `Helixer,StringTie`) |
| `merged_from` | gene | Source gene IDs if this gene was merged from multiple models |
| `exon_posterior` | exon | Per-exon posterior probability |
| `intron_posterior` | intron | Per-intron posterior probability |
| `cds_posterior` | CDS | Per-CDS-segment posterior probability |

---

## Tuning scoring parameters

Run `--dump_config` to write an annotated INI file with all default values:

```bash
python gene_annotation_refiner_2.py --dump_config
# writes: gene_annotation_refiner_default.ini
```

Edit the INI and pass it back with `--config my_config.ini`. The file documents every parameter, which evidence components they affect, and tuning tips. Key parameters to consider:

| Goal | Parameter | Default | Change to |
|---|---|---|---|
| Retain more genes | `coding_threshold` | 0.20 | lower |
| Trust Helixer more | `exon_helixer_support` | 0.85 | raise |
| Penalise unsupported junctions | `intron_bam_score_none` | 0.15 | lower |
| Be lenient without BAM | `intron_bam_score_none` | 0.15 | raise to ~0.40 |
| Require strong junction support | `gene_bam_junction_min_reads` | 2 | raise |

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `numpy` | Yes | Coverage array arithmetic |
| `scipy` | Yes | Smoothing (Savitzky-Golay) for coverage analysis |
| `pyfaidx` | Yes | Random-access genome FASTA |
| `pyBigWig` | Yes | RNA-seq BigWig coverage |
| `pysam` | No | BAM-based junction counting (only with `--bam`) |

---

## Repository layout

```
gene_annotation_refiner_2.py   Main pipeline
test_022362/                   Subset test region (scaffold_2:76722198-77114949)
  make_subset.sh               HPC script to extract subset from full data
  refine_022362.sh             Local run script for the subset
  subset_*.{fa,gff,gtf,bw,tab} Subset input files
test_run/                      Full-genome run scripts and outputs
  refine2.sh                   SLURM batch script for HPC run
```

---

## Citation / contact

Developed for the *Acyrthosiphon pisum* genome annotation project in the Stern Lab (HHMI Janelia Research Campus). If you use this pipeline, please cite the repository URL.
