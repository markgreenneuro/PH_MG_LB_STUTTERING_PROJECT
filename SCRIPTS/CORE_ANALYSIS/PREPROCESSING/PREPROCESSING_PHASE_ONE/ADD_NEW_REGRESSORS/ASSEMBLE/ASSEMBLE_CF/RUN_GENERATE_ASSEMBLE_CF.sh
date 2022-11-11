#!/bin/bash
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=generate_assemble_cf
#SBATCH --time=0-00:05:00
#SBATCH --partition=partition-1
#SBATCH --ntasks=1
#SBATCH --output=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/ASSEMBLE/ASSEMBLE_MATRIX_CF/generate_assemble_cf-%j.output
#SBATCH --error=/home/markgreenneuroscience_gmail_com/LOG/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/ASSEMBLE/ASSEMBLE_MATRIX_CF/generate_assemble_cf-%j.error

wd=/home/markgreenneuroscience_gmail_com/TS_KF_TRANSFORMS/ADD_NEW_REGRESSORS/ASSEMBLE/ASSEMBLE_CF

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 GENERATE_ASSEMBLE_CF.py
