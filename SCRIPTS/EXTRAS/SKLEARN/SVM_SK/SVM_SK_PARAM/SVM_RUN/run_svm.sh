#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -c 6
#SBATCH --mem-per-cpu=1GB
#SBATCH --job-name=svm
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --output=/users/k1754828/LOG/ML/SVM/SVM_RUN/svm-%j.output
#SBATCH --error=/users/k1754828/LOG/ML/SVM/SVM_RUN/svm-%j.error

wd=/users/k1754828/LOG/ML/SVM/SVM_RUN

control_file_dir=/users/k1754828/ML/SVM/SVM_CF
svm_control_file=${control_file_dir}/svm_cf.txt
c=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${svm_control_file} | awk '{print $1}')
kernel=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${svm_control_file} | awk '{print $2}')
gamma=$(sed -n "${SLURM_ARRAY_TASK_ID}"p ${svm_control_file} | awk '{print $3}')
decision_shape_function_shape=$(sed -n "${SGE_TASK_ID}"p ${svm_control_file} | awk '{print $4}')

source /etc/bash.bashrc
eval "$(conda shell.bash hook)"
conda activate python3710

cd $wd || exit
python3 svm.py "$c" "$kernel" "$gamma" "$decision_shape_function_shape"
