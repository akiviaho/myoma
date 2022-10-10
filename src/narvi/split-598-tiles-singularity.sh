#!/bin/bash
#SBATCH --job-name=batch3_4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=antti.kiviaho@tuni.fi
#SBATCH --mail-type=END

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --time=1-23:59:00
#SBATCH --mem=10G  # Total amount of memory reserved for job

cd /lustre/scratch/kiviaho/myoma/myoma-new/tiling-dir

singularity exec --bind /lustre/scratch/kiviaho/myoma/myoma-new/tiling-dir/ segmentation-tiling-env.sif python3.7 -u split-and-segment.py slides_91-108_batch_3.txt

