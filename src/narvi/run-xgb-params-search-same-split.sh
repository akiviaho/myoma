#!/bin/bash
#SBATCH -t 6-23:59:00
#SBATCH -J xgb-params-search
#SBATCH -o xgb-params-search.out.%j
#SBATCH -e xgb-params-search.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu
# :teslav100:1
# #SBATCH --nodelist=nag12
#SBATCH --exclude=meg[10-12],nag[01-09]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"

echo "Starting python job now"

python -u src/xgb-params-search-same-splits.py
