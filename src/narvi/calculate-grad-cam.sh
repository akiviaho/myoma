#!/bin/bash
#SBATCH -t 1-23:59:00
#SBATCH -J grad-cam
#SBATCH -o grad-cam.out.%j
#SBATCH -e grad-cam.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu
# #SBATCH --gres=gpu:teslav100:1
# #SBATCH --nodelist=nag15
# #SBATCH --exclude=meg[10-12],nag[01-09]
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
python -u /lustre/scratch/kiviaho/myoma/myoma-new/src/grad-cam-heatmap-generator.py YEATS4 2000
