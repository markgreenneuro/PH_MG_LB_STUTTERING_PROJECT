#!/bin/bash -l
#SBATCH -c 6
#SBATCH --mem=6GB
#SBATCH --job-name=random_forest
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_RUN/random_forest-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/RANDOM_FOREST/RANDOM_FOREST_RUN/random_forest-%j.error

wd=/users/k1754828/SCRIPTS/RANDOM_FOREST/RANDOM_FOREST_RUN

control_file_dir=/users/k1754828/SCRIPTS/RANDOM_FOREST/RANDOM_FOREST_CF
random_forest_control_file=${control_file_dir}/random_forest_cf.txt
splits=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${random_forest_control_file} | awk '{print $1}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 random_forest.py "$splits"
