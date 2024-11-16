from pre_dataset2 import QwenDataset, collate_fn
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from functools import partial

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")
dataset = QwenDataset("./table-extraction.json")

print(len(dataset))

train_loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=partial(collate_fn, processor=processor, max_len=-1),
)

print(len(train_loader))

sizes = []
for i, batch in enumerate(train_loader):
    inp, label = batch
    print(label.shape)
    sizes.append(label.shape[-1])

print(f"Maximum sequence length: {max(sizes)}")
print(f"Average sequence length: {sum(sizes) / len(sizes):.2f}")
print(f"Minimum sequence length: {min(sizes)}")
print(f"Number of sequences > 2048: {sum(1 for s in sizes if s > 2048)}")