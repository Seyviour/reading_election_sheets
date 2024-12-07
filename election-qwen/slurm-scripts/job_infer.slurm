#! /bin/bash
# ====================================
#SBATCH --job-name=qwen_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=23:59:58
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
#SBATCH --output=qwen_infer_output.txt
#SBATCH --error=qwen_infer_error.txt
# ====================================

eval "$(~/software/miniforge3/bin/conda shell.bash hook)"

conda activate pytorch2

python -u /home/saviour.owolabi/enel645/final_project/qwen_infer.py
