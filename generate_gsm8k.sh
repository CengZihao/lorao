#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export LD_LIBRARY_PATH=/data/zzh/d2l/miniconda3/envs/O-LoRA/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="7"

# bash generate.sh

version="0115_gsm8k_lorao"

# /data/zzh/d2l/miniconda3/envs/lora/bin/python -u /data/zzh/olora/eval_gsm8k_tra.py \
#     --lora_weights "/data/zzh/olora/lora-alpaca/0113_v2_gsm8k_tra" \
#     > log/generate/$version.log 2>&1

/data/zzh/d2l/miniconda3/envs/lorao/bin/python -u /data/zzh/olora/eval_gsm8k_lorao.py \
    --lora_weights "/data/zzh/olora/lora-alpaca/0114_v6_gsm8k_tau1" \
    > log/generate/$version.log 2>&1