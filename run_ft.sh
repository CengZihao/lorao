#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export LD_LIBRARY_PATH=/data/zzh/d2l/miniconda3/envs/O-LoRA/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"

# bash run_ft.sh

version="0115_v3_tsa_tau-15"

echo -e "==================================================\n" \
    "$version\n" \
    "tsa tau -15\n" \
    "==================================================\n\n" \
    > log/$version.log

/data/zzh/d2l/miniconda3/envs/lorao/bin/python /data/zzh/olora/finetune.py \
    --output_dir "./lora-alpaca/$version" \
    --resume_from_checkpoint "/data/zzh/olora/lora-alpaca/0115_v1_piqa_tau-2" \
    >> log/$version.log 2>&1

# /data/zzh/d2l/miniconda3/envs/lorao/bin/python /data/zzh/olora/finetune.py \
#     --output_dir "./lora-alpaca/$version" \
#     >> log/$version.log 2>&1

# /data/zzh/d2l/miniconda3/envs/lora/bin/python /data/zzh/olora/finetune_traditional.py \
#     --output_dir "./lora-alpaca/$version" \
#     >> log/$version.log 2>&1

# /home/zzh/miniconda3/envs/lorao/bin/python /home/zzh/olora/finetune.py \
#     --output_dir "./lora-alpaca/$version" \
#     --resume_from_checkpoint "/home/zzh/olora/lora-alpaca/1219_v3" \
#     >> log/$version.log 2>&1