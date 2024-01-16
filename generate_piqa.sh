#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export LD_LIBRARY_PATH=/data/zzh/d2l/miniconda3/envs/O-LoRA/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="1"

version="0115_piqa_lorao"

/data/zzh/d2l/miniconda3/envs/lorao/bin/python -u /data/zzh/olora/eval_piqa_lorao.py \
    --lora_weights "/data/zzh/olora/lora-alpaca/0115_v1_piqa_tau-2" \
    > log/generate/$version.log 2>&1