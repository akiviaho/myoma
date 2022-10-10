#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J 5-fold-confident-clustering
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=700G
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/10.0
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-env
echo "conda activated"

python -u src/cluster-confident-tiles.py
