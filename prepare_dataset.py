from PIL import Image
from io import BytesIO
from base64 import b64decode
from datasets import Dataset as HDataset
from torch.utils.data import Dataset
# from transformers import AutoProcessor

class QwenDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        print("Loading data...")

        self.dataset = HDataset.from_json("/workspace/data.json")

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
                "role": 'asisstant',
                'content': sample["response"]
            }
        ]
        return self.processor.apply_chat_template(msg)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text_prompt = self.make_template(data)
        img = Image.open(BytesIO(b64decode(data["image"]))).convert('RGB')
        inputs = self.processor(text=[text_prompt], images=[img], return_tensors="pt")
        return inputs


# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# ds = QwenDataset(
#     processor
# )
# print(len(ds))
# print(ds[0]['input_ids'].shape)