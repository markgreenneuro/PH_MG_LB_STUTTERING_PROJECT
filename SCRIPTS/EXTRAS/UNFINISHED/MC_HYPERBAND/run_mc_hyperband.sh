#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=mc_hh
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/HYPERBANDING/MC_HYPERBAND/mc_hyperband-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/HYPERBANDING/MC_HYPERBAND/mc_hyperband-%j.error

wd=/users/k1754828/SCRIPTS/HYPERBANDING/MC_HYPERBAND


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3510

cd $wd || exit
python3 main.py
