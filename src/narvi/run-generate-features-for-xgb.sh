#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J gen-xgb
#SBATCH -o gen-xgb.out.%j
#SBATCH -e gen-xgb.err.%j
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"


python -u src/generate-features-for-xgb.py