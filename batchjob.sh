#!/bin/bash

#SBATCH -J uni_data
#SBATCH -o outfiles/uni_data_out.log
#SBATCH -e errfiles/uni_data_err.log
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 2:00:00
#SBATCH -A OTH21076
#SBATCH --mail-user=bcchang@utexas.edu
#SBATCH --mail-type=all

#source /work/06898/bchang/ls6/ms_net_mostcurrent1/ms-net/bin/activate
#source /work/06898/bchang/ls6/miniconda3/etc/profile.d/conda.sh
#conda activate fl


CUDA_VISIBLE_DEVICES=0 python server.py device="cuda" test_input_file=server_tiff_net.yml data_loc=$SCRATCH/data num_rounds=100 &
sleep 5
CUDA_VISIBLE_DEVICES=1 python client.py device="cuda" train_input_file=client_tiff_net.yml data_loc=$SCRATCH/data &
sleep 5
CUDA_VISIBLE_DEVICES=2 python client.py device="cuda" train_input_file=client_tiff_net.yml data_loc=$SCRATCH/data &
wait
