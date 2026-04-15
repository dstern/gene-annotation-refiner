#!/bin/bash
#SBATCH --job-name=Apis_refine_gff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --error=refine.err
#SBATCH --output=refine.out

Acyrthosiphon_pisum_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum'
JICv1_0_mapped_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum/Annotate/JICv1.0_mapped'
Apis_annotate_4May2025_folder='/home/ds2996/sternlab/Lab_Data/genome_data/Acyrthosiphon_pisum/Annotate/JICv1.0_mapped/Apis_annotate_4May2025'

# conda create -n refine_annot_env -c conda-forge -c bioconda python=3.10 numpy scipy pyfaidx pybigwig pysam

ml conda
conda activate refine_annot_env

# Note: use the .tab junction file (not .bed) — it contains actual read counts
# (nb_raw_aln column) needed for evidence calibration. The .bed file only has
# Portcullis confidence scores, which are not useful as read counts.
# Note: inline bash comments after a backslash (\ #comment) break line
# continuation — keep comments on separate lines as shown here.

/home/ds2996/sternlab/bin/gene_annotation_refiner_2.py \
    --genome "$Acyrthosiphon_pisum_folder/Genomes/v1.0_JIC/Acyrthosiphon_pisum_JIC1_v1.0.scaffolds_mtDNA.fa" \
    --output "${JICv1_0_mapped_folder}/Apis_refined_updated_v2.4_13iv26.gff" \
    --helixer "$Apis_annotate_4May2025_folder/Apis_helixer_3v25.sorted.gff" \
    --stringtie "$Apis_annotate_4May2025_folder/Apis.mixed.f0.75.gtf" \
    --transdecoder "$JICv1_0_mapped_folder/Acyrthosiphon_pisum.27xi23_isoseq.transdecoder_isoseqUpdated.updated.12xii23.gff3" \
    --bigwig "$JICv1_0_mapped_folder/Apis.bam.bw" \
    --junctions "$Apis_annotate_4May2025_folder/portcullis_out_all/3-filt/portcullis_filtered.pass.junctions.tab" \
    --renumber \
    --gene_prefix Apis_refined_13iv26

#   --transdecoder "$Apis_annotate_4May2025_folder/Apis.mixed.f0.75.transcripts.fasta.TD2.genome.gff3"
#   --bam "$Apis_annotate_4May2025_folder/Apis.all.merged.bam"
#   --config "$JICv1_0_mapped_folder/config_jct08.ini"

conda deactivate
