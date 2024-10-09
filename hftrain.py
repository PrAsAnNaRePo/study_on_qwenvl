from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Any, Dict, Optional, List, Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pre_dataset2 import QwenDataset, collate_fn, partial
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-2B-Instruct")


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="./data.json", metadata={"help": "Path to the training data."}
    )
    max_len: int = field(
        default=3500, metadata={"help": "Max length of the text."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

class QwenTrainer(Trainer):
    def compute_loss(self, model, inputs, labels, return_outputs=False):
        outputs = model(**inputs, label=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

     
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    model = Vamos(config=config)
    train_set = LLavaDataset(
        model.processor,
        model.tokenizer,
        **data_args.__dict__,
    )
    
    trainer = QwenTrainer(
        model=model, args=training_args, train_dataset=train_set, process=
    )
    trainer.train(resume_from_checkpoint=True if training_args.resume_from_checkpoint else None)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
