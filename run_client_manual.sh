#!/bin/bash

#source /work/06898/bchang/ls6/ms_net_mostcurrent1/ms-net/bin/activate
#source /work/06898/bchang/ls6/miniconda3/etc/profile.d/conda.sh
#conda activate fl

round_num=9

CUDA_VISIBLE_DEVICES=0 python client_manual.py device="cuda" train_input_file=client_tiff_net.yml model_dir=$STOCKYARD/fl/client_models/ut round=$round_num &
sleep 5
CUDA_VISIBLE_DEVICES=1 python client_manual.py device="cuda" train_input_file=client_tiff_net.yml model_dircase 
=$STOCKYARD/fl/client_models/petrobras round=$round_num &
sleep 5
CUDA_VISIBLE_DEVICES=2 python client_manual.py device="cuda" train_input_file=client_tiff_net.yml model_dir=$STOCKYARD/fl/client_models/bp round=$round_num &
wait
