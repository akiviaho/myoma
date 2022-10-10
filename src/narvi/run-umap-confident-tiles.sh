#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J 5-fold-umap
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=38
#SBATCH --mem=750G
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=antti.kiviaho@tuni.fi

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"

python -u src/umap-confident-tiles.py fold_5