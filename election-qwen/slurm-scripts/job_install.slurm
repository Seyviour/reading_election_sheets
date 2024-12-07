#! /bin/bash
# ====================================
#SBATCH --job-name=qwen_finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=23:59:58
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
#SBATCH --output=qwen_output.txt
#SBATCH --error=qwen_error.txt
# ====================================

eval "$(~/software/miniforge3/bin/conda shell.bash hook)"

conda activate pytorch2

#conda install -c conda-forge pytorch=2.1.2 python pip
#conda install -c conda-forge torchvision cuda-nvcc
#pip install unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git