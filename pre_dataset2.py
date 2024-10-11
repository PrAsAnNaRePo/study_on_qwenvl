from PIL import Image
from io import BytesIO
from base64 import b64decode
from datasets import Dataset as HDataset
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from tqdm import tqdm
import json
from qwen_vl_utils import process_vision_info
from functools import partial

class QwenDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = HDataset.from_json(data_path)

        self.conversation_data = []
        for i in self.data:
            self.conversation_data.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract the table in this image."},
                                {"type": "image", "image": f"data:image;base64,{i['image']}"},
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": i['response']}
                            ]
                        }
                    ]
                }
            )

    def __len__(self):
        return len(self.conversation_data)

    def __getitem__(self, idx):
        return self.conversation_data[idx]

def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    for i in range(len(l) - 1):
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, max_len):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    if max_len == -1:
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=max_len,
            truncation=True
        )

    # inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")
# dataset = QwenDataset("/workspace/study_on_qwenvl/data.json")

# print(len(dataset))

# train_loader = DataLoader(
#     dataset,
#     batch_size=1,
#     collate_fn=partial(collate_fn, processor=processor, device="cuda")
# )

