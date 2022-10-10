#!/bin/bash
#SBATCH -t 06-23:59:00
#SBATCH --partition=normal
# #SBATCH --nodelist=me233,me2434,me234,me236,me237,me238,me239
#SBATCH -J f5-package
#SBATCH -o f5-package.out.%j
#SBATCH -e f5-package.err.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"

python -u src/package-to-tfr.py \
    --tiles /lustre/scratch/kiviaho/myoma/myoma-new/data/tile_split/fold-5-in-5-tile-paths.tsv \
    --batch 1600 \
    --identifier fold_5_ \
    --save_path /lustre/nvme/kiviaho/tfrecords/ \
    --seed_number 42 \
    --jobid ${SLURM_JOBID}