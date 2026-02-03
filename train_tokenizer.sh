#!/bin/bash
#SBATCH --job-name=d40tokenizer
#SBATCH --partition=NV_H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/public_hw/share/cit_ztyu/cz/nanochat/logs/tokenizer_%j.log
#SBATCH --error=/public_hw/share/cit_ztyu/cz/nanochat/logs/tokenizer_%j.err

cd /public_hw/share/cit_ztyu/cz/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR=/public_hw/share/cit_ztyu/cz/nanochat
export PYTHONPATH=$PYTHONPATH:/public_hw/share/cit_ztyu/cz/nanochat

python -m scripts.tok_train