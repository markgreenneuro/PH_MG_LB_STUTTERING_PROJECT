#!/bin/bash -l
#SBATCH -c 10
#SBATCH --mem=10G
#SBATCH --job-name=log_reg_l1_rmsprop_net_params
#SBATCH --time=0-48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/LR/PARAMS/LR_NN_PARAMS_L1/LR_NN_L1_RMSPROP_PARAM/log_reg_net_l1_rmsprop_params-%j.output
#SBATCH --error=/users/k1754828/LOG/CORE_ANALYSIS/MODELLING/LR/PARAMS/LR_NN_PARAMS_L1/LR_NN_L1_RMSPROP_PARAM/log_reg_net_l1_rmsprop_params-%j.error

wd=/users/k1754828/SCRIPTS/CORE_ANALYSIS/MODELLING/LR/PARAMS/LR_NN_PARAMS_L1/LR_NN_L1_RMSPROP_PARAM


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 main_reopen.py

