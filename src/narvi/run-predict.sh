#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J predict-1-miss
#SBATCH -o predict-1-miss.out.%j
#SBATCH -e predict-1-miss.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
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
# run with 1 GPU and a trained model
# SWITCH BETWEEN FLAGS:
# --binary_model
# --multiclass_model
# These flags do not actually matter since the number of possible labels is already in the model.
python src/predict.py  \
  --model multiclass_fold_1_missing_2_epochs_100_percent_21939635_at_2022-07-15_21:45:39_598px.h5 \
  --tfrecords /lustre/nvme/kiviaho/fold_1_tfrs.tsv \
  --multiclass_model \
  --img_width 598 \
  --img_height 598 \
  --batch 8