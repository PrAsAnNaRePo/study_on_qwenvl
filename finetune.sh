#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="nnpy/adeos-qwen2.5-2b-v1"
DATA="ft-adeos-2562.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS qwen_ft.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --fix_vit True \
    --output_dir adeos-qwen2.5-2b-v1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 3200 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ds_config_zero3.json \
    --push_to_hub True