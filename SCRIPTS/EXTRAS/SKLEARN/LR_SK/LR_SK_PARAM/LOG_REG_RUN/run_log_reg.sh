#!/bin/bash -l
#SBATCH -c 6
#SBATCH --mem=59000
#SBATCH --job-name=log_reg
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --output=/users/k1754828/LOG/ML/LOG_REG/LOG_REG_RUN/log_reg_run-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/LOG_REG/LOG_REG_RUN/log_reg_run-%j.error

wd=/users/k1754828/ML/LOG_REG/LOG_REG_RUN

control_file_dir=/home/markgreenneuroscience_gmail_com/ML/LOG_REG/LOG_REG/LOG_REG_ONE_CF
log_reg_control_file=${control_file_dir}/LOG_REG_ONE_CF.txt
penalty_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $1}')
c_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $2}')
solver_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $3}')
multi_class_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $4}')
elastic_net_term=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${log_reg_control_file} | awk '{print $5}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate myenv

cd $wd || exit
python3 LOG_REG.py "$penalty_term" "$c_term" "$solver_term" "$multi_class_term" "$elastic_net_term"
