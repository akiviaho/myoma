#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J dset-split-and-package
#SBATCH -o split-and-package.out.%j
#SBATCH -e split-and-package.err.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36000
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/10.0
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-env
echo "conda activated"


