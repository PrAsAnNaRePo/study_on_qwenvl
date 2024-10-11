#!/bin/bash

if ! command -v deepspeed &> /dev/null
then
    echo "DeepSpeed could not be found. Please install it before running this script."
    exit
fi

NUM_GPUS=4
MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
MAX_LEN=3500
DATA_PATH="./data.json"
EPOCHS=7
DEEPSPEED_CONFIG="ds_config.json"

echo "Running training with the following configuration:"
echo "Number of GPUs: $NUM_GPUS"
echo "Model ID: $MODEL_ID"
echo "Max Length: $MAX_LEN"
echo "Data Path: $DATA_PATH"
echo "Epochs: $EPOCHS"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"

deepspeed --num_gpus=$NUM_GPUS train.py --model_id $MODEL_ID --max_len $MAX_LEN --data_path $DATA_PATH --epochs $EPOCHS --deepspeed_config $DEEPSPEED_CONFIG
