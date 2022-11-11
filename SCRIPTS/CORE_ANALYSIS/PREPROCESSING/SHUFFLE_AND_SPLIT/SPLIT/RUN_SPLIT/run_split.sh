#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10GB
#SBATCH --job-name=split
#SBATCH --time=0-00:30:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/SPLIT/RUN_SPLIT_MATRIX/split_matrix-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/SPLIT/RUN_SPLIT_MATRIX/split_matrix-%j.error

wd=/users/k1754828/SCRIPTS/SHUFFLE_AND_SPLIT/SPLIT/RUN_SPLIT

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 split.py

