#!/bin/bash
#SBATCH -t 1-23:59:00
#SBATCH -J extract-fold-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --exclude=meg[10-12],nag[01-09]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate myoma-new-env
echo "conda activated"

echo "Starting python job now"

python -u src/extract-conv2d_93-from-confident.py \
    --predictions_file multiclass_fold_4_missing_2_epochs_100_percent_21939632_at_2022-07-15_16:04:26_598px_multiclass_prediction_results.csv \
    --model_file multiclass_fold_4_missing_2_epochs_100_percent_21939632_at_2022-07-15_16:04:26_598px.h5 \
    --sample_info_file fold_4_tfr_contents.tsv \
    --save_metadata_file /lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/test_confidently_predicted.tsv \
    --save_conv_layer_file /lustre/scratch/kiviaho/myoma/myoma-new/conv2d_93_layers/test_conv2d_93_layers.npy
    