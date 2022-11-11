#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=mlp_four_hyperband
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/MLP/MLP_FOUR/mlp_four_hyperband-%j.output
#SBATCH --error=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/MLP/MLP_FOUR/mlp_four_hyperband-%j.error

wd=/users/k1754828/SCRIPTS/CORE_ANALYSIS/MODELLING/MLP/MLP_FOUR


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main.py

