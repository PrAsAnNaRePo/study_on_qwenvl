import argparse
import json
import deepspeed
import requests
from PIL import Image
from tqdm import tqdm
from PIL import Image
import torch
from io import BytesIO
from base64 import b64decode
import numpy as np
# from prepare_dataset import QwenDataset
from pre_dataset2 import QwenDataset, collate_fn
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from functools import partial

def add_argument():

    parser=argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max_len", type=int, default=5_000)
    parser.add_argument("--data_path", type=str, default="./data.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ds_config", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    
    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args=parser.parse_args()

    return args


def initialize(args,
               model,
               processor,
               optimizer=None,
               parameters=None,
               training_data=None,
               ):
    parameters = filter(lambda p: p.requires_grad, model.parameters()) if parameters is None else parameters
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=training_data, collate_fn=partial(collate_fn, processor=processor, max_len=args.max_len))
    return model_engine, optimizer, trainloader

def train(args, model_engine, train_loader):
    # Training loop.
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, leave=False)
        for batch_idx, batch in enumerate(loop):
            inputs, labels = batch
            inputs = inputs.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)

            loss = model_engine(**inputs, labels=labels).loss
            model_engine.backward(loss)
            model_engine.step()
        model_engine.save_checkpoint(f"checkpoints")

def main():
    args = add_argument()
    
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="cpu",
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model_id)
    training_data = QwenDataset(args.data_path)

    model_engine, optimizer, train_loader = initialize(
        args=args,
        model=model,
        processor=processor,
        training_data=training_data,
    )
    train(args, model_engine, train_loader)

if __name__ == "__main__":
    main()
