#!/bin/bash -l
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=generate_random_forest_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --output/users/k1754828/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_CF/generate_random_forest_cf-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_CF/generate_random_forest_cf-%j.error

wd=/users/k1754828/SCRIPTS/RANDOM_FOREST/RANDOM_FOREST_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 generate_random_forest_cf.py
