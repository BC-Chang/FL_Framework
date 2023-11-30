#!/bin/bash

#SBATCH -J fedProx
#SBATCH -o outfiles/fedProx_out.log
#SBATCH -e errfiles/fedProx_err.log
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 24:00:00
#SBATCH -A OTH21076
#SBATCH --mail-user=bcchang@utexas.edu
#SBATCH --mail-type=all

#source /work/06898/bchang/ls6/ms_net_mostcurrent1/ms-net/bin/activate
source /work/06898/bchang/ls6/miniconda3/etc/profile.d/conda.sh
conda activate fl


CUDA_VISIBLE_DEVICES=0 python server.py device="cuda" test_input_file=test_msnet.yml data_loc=$SCRATCH/data num_rounds=50 config_fit.local_epochs=10 strategy=fedprox &
CUDA_VISIBLE_DEVICES=1 python client.py device="cuda" train_input_file=train_val_msnet.yml data_loc=$SCRATCH/data &
CUDA_VISIBLE_DEVICES=2 python client.py device="cuda" train_input_file=train_val_msnet.yml data_loc=$SCRATCH/data &

wait
