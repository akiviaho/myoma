#!/bin/bash
#SBATCH --job-name=convert-sif
#SBATCH --output=slurm-convert-%j.out
#SBATCH --error=slurm-convert-%j.err
#SBATCH --mail-user=antti.kiviaho@tuni.fi
#SBATCH --mail-type=END

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G  # Total amount of memory reserved for job

INPUTDOCKER=$1
OUTPUTSIF=$2

# https://docs.csc.fi/computing/containers/run-existing/

# Let's use the fast local drive for temporary storage
export SINGULARITY_TMPDIR=$LOCAL_SCRATCH
export SINGULARITY_CACHEDIR=$LOCAL_SCRATCH

# This is just to avoid some annoying warnings
unset XDG_RUNTIME_DIR

singularity build ${OUTPUTSIF} docker-archive://${INPUTDOCKER}