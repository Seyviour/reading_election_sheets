#! /bin/bash
# ====================================
#SBATCH --job-name=tl_image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --time=23:59:58
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
# ====================================

eval "$(~/software/miniforge3/bin/conda shell.bash hook)"

conda activate pytorch

python -u  /home/ziheng.chang/final_project/paligemma_vote.py
