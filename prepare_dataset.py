from PIL import Image
from io import BytesIO
from base64 import b64decode
from datasets import Dataset as HDataset
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from tqdm import tqdm

class QwenDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        print("Loading data...")

        self.dataset = HDataset.from_json("/workspace/data.json")
        self.processed_data = self.preprocess_dataset()

    def preprocess_dataset(self):
        processed_data = []
        for data in tqdm(self.dataset, desc="Processing dataset"):
            text_prompt = self.make_template(data)
            img = Image.open(BytesIO(b64decode(data["image"]))).convert('RGB')
            w, h = img.size
            if w < 28 or h < 28:
                img = img.resize((28 if w < 28 else w, 28 if h < 28 else h))
                print("resized: ", img.size)
            
            inputs = self.processor(
                text=[text_prompt], 
                images=[img],
                return_tensors="pt", 
                # max_length=5000, 
                # padding="max_length", 
                # truncation=True
            )

            processed_data.append({
                "input_ids": inputs['input_ids'],
                "pixel_values": inputs["pixel_values"],
                'attention_mask': inputs['attention_mask'],
                "image_grid_thw": inputs['image_grid_thw']
            })

        return processed_data

    def make_template(self, sample):
        msg = [
            {
                "role": 'user',
                'content': [
                    {"type": "text", "text": "Extract the table from this image"},
                    {
                        "type": "image",
                    },
                ],
            },
            {
                "role": 'assistant',
                'content': sample["response"]
            }
        ]
        return self.processor.apply_chat_template(msg)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        return self.processed_data[index]

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, # Binary floating point helps to load the model in lower precision with lower memory size.
    attn_implementation="flash_attention_2", # Flash attention helps to reduce memory usage as the context length increase.
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

ds = QwenDataset(
    processor
)

print(len(ds))
print(ds[0]['input_ids'].shape)
print(ds[0]['pixel_values'].shape)
print(ds[0]['attention_mask'].shape)
print(ds[0]['image_grid_thw'].shape)

data = {
    "input_ids": ds[0]['input_ids'].cuda(),
    "attention_mask": ds[0]['attention_mask'].cuda(),
    "pixel_values": ds[0]['pixel_values'].cuda(),
    'image_grid_thw': ds[0]['image_grid_thw'].cuda()
}

with torch.cuda.amp.autocast_mode.autocast():
    out = model(**data)