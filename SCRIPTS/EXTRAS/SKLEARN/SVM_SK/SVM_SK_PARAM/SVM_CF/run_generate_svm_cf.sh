#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=1GB
#SBATCH --job-name=generate_svm_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/SCRIPTS/LOG/ML/SVM/SVM_CF/generate_svm_cf-%j.output
#SBATCH --error=/users/k1754828/SCRIPTS/LOG/ML/SVM/SVM_CF/generate_svm_cf-%j.error

wd=/users/k1754828/SCRIPTS/SVM/SVM_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 generate_svm_cf.py
