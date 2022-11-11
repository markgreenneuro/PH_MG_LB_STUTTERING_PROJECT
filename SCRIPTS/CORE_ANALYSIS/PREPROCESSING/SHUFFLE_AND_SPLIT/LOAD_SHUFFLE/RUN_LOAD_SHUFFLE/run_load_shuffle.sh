#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=load_shuffle
#SBATCH --time=0-01:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/LOAD_SHUFFLE/load_shuffle-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/LOAD_SHUFFLE/load_shuffle-%j.error

wd=/users/k1754828/SCRIPTS/LOAD_SHUFFLE/RUN_LOAD_SHUFFLE

seed_value=1234
root='/scratch/users/k1754828/DATA/'
file_name='master2.csv'


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 load_shuffle.py "$seed_value" "$root" "$file_name"
