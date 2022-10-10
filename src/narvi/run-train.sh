#!/bin/bash
#SBATCH -t 6-23:59:00
#SBATCH -J train-1-missing
#SBATCH -o train-1-missing.out.%j
#SBATCH -e train-1-missing.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
# #SBATCH --nodelist=nag16
#SBATCH --exclude=meg[10-12],nag[01-09]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36000
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"

echo "Starting python job now"
# train the network
# SWITCH BETWEEN FLAGS:
# --binary_model
# --multiclass_model
python -u /lustre/scratch/kiviaho/myoma/myoma-new/src/train.py \
        --multiclass_model \
        --tfrecords  /lustre/nvme/kiviaho/folds_2_3_4_5_tfrs.tsv  \
        --sample_sheet /lustre/scratch/kiviaho/myoma/myoma-new/fold_1_missing_tfr_contents.csv \
        --prefix fold_1_missing_ \
        --no_validation \
        --seed_number 25 \
        --img_width 598  \
        --img_height 598  \
        --size_tfrecord 1600 \
        --percent 100 \
        --epochs 2  \
        --learning_rate 0.0001 \
        --batch  8 \
        --jobid ${SLURM_JOBID}
