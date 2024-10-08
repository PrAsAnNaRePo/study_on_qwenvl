import argparse
import json
import deepspeed
import requests
from PIL import Image
from dotmap import DotMap
from tqdm import tqdm
from PIL import Image
import torch
from io import BytesIO
from base64 import b64decode
import numpy as np
from datasets import load_dataset, concatenate_datasets
from prepare_dataset import QwenDataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def add_argument():

    parser=argparse.ArgumentParser()

    parser.add_argument("--ds_config", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    
    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args=parser.parse_args()

    return args


def initialize(args,
               model,
               optimizer=None,
               parameters=None,
               training_data=None,
               ):
    parameters = filter(lambda p: p.requires_grad, model.parameters()) if parameters is None else parameters
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=training_data)
    return model_engine, optimizer, trainloader

def train(args, model_engine, train_loader):
    # Training loop.
    for epoch in range(args.training.epochs):
        loop = tqdm(train_loader, leave=False)
        for batch_idx, batch in enumerate(loop):
            pixels = batch[0].reshape(-1, 1, 3, 224, 224).to(model_engine.local_rank)
            input_ids = batch[1].reshape(-1, args.data.max_len).to(model_engine.local_rank)

            loss = model_engine(input_ids, pixels, True).loss
            model_engine.backward(loss)
            model_engine.step()
        model_engine.save_checkpoint(f"checkpoints")

def main():
    args = add_argument()
    
    config_file = 'config.json'
    with open(config_file) as f:
        config = DotMap(json.load(f))
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.llm_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(config.llm_id)
    training_data = QwenDataset(processor)

    model_engine, optimizer, train_loader = initialize(
        args=args,
        model=model,
        training_data=training_data,
    )
    train(config, model_engine, train_loader)

if __name__ == "__main__":
    main()
