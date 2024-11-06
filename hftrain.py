from dataclasses import dataclass, field
import os
from typing import Optional
import torch
from torch import nn
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import HfArgumentParser, TrainingArguments
from pre_dataset2 import QwenDataset, partial, collate_fn
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.utils.data import DataLoader

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-2B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(
        default="./data.json", metadata={"help": "Path to the training data."}
    )
    max_len: int = field(
        default=3500, metadata={"help": "Max length of the text."}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    num_train_epochs: int = field(default=7)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed config file."})

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def rank0_print(local_rank, *args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict() if trainer.is_deepspeed_enabled else trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

class QwenTrainer(Trainer):
    def __init__(self, *args, processor=None, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.data_args = data_args  # Store data_args as an instance attribute

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs, labels = inputs
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=partial(
                collate_fn,
                processor=self.processor,
                max_len=self.data_args.max_len  # Use self.data_args.max_len here
            ),
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # accelerator = Accelerator()
    
    if training_args.deepspeed_config:
        training_args.deepspeed = training_args.deepspeed_config
        # training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    processor = transformers.AutoProcessor.from_pretrained(model_args.model_id)

    train_set = QwenDataset(data_args.data_path)
    
    trainer = QwenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        processor=processor,
        data_args=data_args,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()