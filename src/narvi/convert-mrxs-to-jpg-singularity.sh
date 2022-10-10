#!/bin/bash
#SBATCH --job-name=test_slides
# #SBATCH --output=slurm-%j.out
# #SBATCH --error=slurm-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --time=1-23:59:00
#SBATCH --mem=60G  # Total amount of memory reserved for job

cd /lustre/scratch/kiviaho/myoma/myoma-new/tiling-dir

singularity exec --bind /lustre/scratch/kiviaho/myoma/myoma-new/tiling-dir/ segmentation-tiling-env.sif python3.7 -u convert-mrxs-to-jpg.py Slides-2019-HE-JPG-16-downsample slides-old.txt
