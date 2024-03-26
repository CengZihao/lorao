#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1,4,7"

version="0325_v4_task1"

echo -e "==================================================\n" \
    "$version\n" \
    "tsa\n" \
    "==================================================\n\n" \
    > log/$version.log

/data/zzh/d2l/miniconda3/envs/lorao/bin/python /data/zzh/olora/finetune_0316.py \
    --output_dir "./lora-alpaca/$version" \
    --select_dataset "tsa" \
    >> log/$version.log 2>&1

# /data/zzh/d2l/miniconda3/envs/lorao/bin/python /data/zzh/olora/finetune_0316.py \
#     --output_dir "./lora-alpaca/$version" \
#     --resume_from_checkpoint "/data/zzh/olora/lora-alpaca/0325_v3_task1" \
#     --select_dataset "boolq" \
#     >> log/$version.log 2>&1
