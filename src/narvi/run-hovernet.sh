#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH -J hovernet
#SBATCH -o hovernet.out.%j
#SBATCH -e hovernet.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --exclude=meg[10-12],nag[01-09]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=350G
#SBATCH --mail-type=END
#SBATCH --mail-user=antti.kiviaho@tuni.fi

module load CUDA/11.2
echo "CUDA loaded"

# load conda environment
module load anaconda
source activate hovernet
echo "conda activated"

echo "Starting python job now"

python hovernet/hover_net/run_infer.py \
  --model_path=/lustre/scratch/kiviaho/myoma/myoma-new/hovernet/hovernet_original_consep_notype_pytorch.tar \
  --model_mode='original' \
  tile \
  --input_dir=/lustre/scratch/kiviaho/myoma/myoma-new/hovernet/data/HMGA1_cluster_2_confident_tiles_subsample \
  --output_dir=/lustre/scratch/kiviaho/myoma/myoma-new/hovernet/results/HMGA1_cluster_2