#!/bin/bash

#SBATCH -J MGN_1-0
#SBATCH -o outfiles/MGN_1-0_v2.log
#SBATCH -e errfiles/MGN_1-0_v2.log
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 00:20:00
#SBATCH -A OTH21076
#SBATCH --mail-user=bcchang@utexas.edu
#SBATCH --mail-type=none

#source /work/06898/bchang/ls6/ms_net_mostcurrent1/ms-net/bin/activate
source /work/06898/bchang/ls6/miniconda3/etc/profile.d/conda.sh
conda activate fl


CUDA_VISIBLE_DEVICES=0 python server.py device="cuda" test_input_file=test_msnet.yml data_loc=$SCRATCH/data num_rounds=300 num_clients=1 num_clients_per_round_fit=1 num_clients_per_round_eval=1 dp.use=true dp.noise_multiplier=0 dp.max_grad_norm=100 dp.poisson_sampling=false dp.target_epsilon=null &
CUDA_VISIBLE_DEVICES=1 python client.py device="cuda" train_input_file=train_val_msnet.yml data_loc=$SCRATCH/data dp.use=true dp.noise_multiplier=0 dp.max_grad_norm=100 dp.poisson_sampling=false dp.target_epsilon=null &


wait
