#!/bin/bash
#SBATCH --job-name=name
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER_EMAIL

module purge
module load 2023
module load CUDA/12.1.1

cd USER_DIRECTORY
source .venv/bin/activate

python ./PATH_TO_FILE/src/flex_train_jm.py ./PATH_TO_FILE/configs/flex_training_jm.toml
