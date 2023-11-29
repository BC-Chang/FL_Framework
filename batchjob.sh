#!/bin/bash

#SBATCH -J uni_data
#SBATCH -o outfiles/uni_data_out.log
#SBATCH -e errfiles/uni_data_err.log
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 8:00:00
#SBATCH -A OTH21076
#SBATCH --mail-user=bcchang@utexas.edu
#SBATCH --mail-type=all

#source /work/06898/bchang/ls6/ms_net_mostcurrent1/ms-net/bin/activate
source /work/06898/bchang/ls6/miniconda3/etc/profile.d/conda.sh
conda activate fl


CUDA_VISIBLE_DEVICES=0 python server.py device="cuda:0" test_input_file=test_data.yml data_loc=$SCRATCH/data num_rounds=1000 &
CUDA_VISIBLE_DEVICES=1 python client.py device="cuda:0" train_input_file=train_data.yml data_loc=$SCRATCH/data &
CUDA_VISIBLE_DEVICES=2 python client.py device="cuda:0" train_input_file=train_data.yml data_loc=$SCRATCH/data &

wait
