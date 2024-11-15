'''

`qwen_data_prep.py` helps to prepare the data for fine-tuning in the specific format mentioned in [QwenLM github](https://github.com/QwenLM/Qwen-VL/tree/master)

here is the sample format to make:

[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  },
  { 
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/mm_tutorial/Chongqing.jpeg</img>\nPicture 2: <img>assets/mm_tutorial/Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]

'''

import json
import os
import sys
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from base64 import b64decode
import random

FILE_PATH = "data.json"
IMAGE_DIR = "images"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# load the dataset we have
with open("./data.json", "r") as f:
    data = json.load(f)

print(data.keys()) # dict_keys(['time', 'file_name', 'page_num', 'image', 'response'])

randomized_user_prompt = [
    "Please extract the table from this image",
    "Extract the table shown in the image",
    "Go through the image and extract the table data",
    "Extract the table that is in this image",
    "From the image, please extract the table.",
    "Kindly extract the table present in the image.",
    "Please go through the image and extract its table.",
    "Extract the table contained within the image."
]

qwen_data = []

for i in tqdm(range(len(data['time']))):

    time = data['time'][i]
    file_name = data['file_name'][i]
    page_num = data['page_num'][i]

    # save the image wit unique name
    base64_image = data['image'][i]
    image = Image.open(BytesIO(b64decode(base64_image)))
    image.save(os.path.join(IMAGE_DIR, f"{file_name}-{page_num}-{i}.png"))

    user_query = random.choice(randomized_user_prompt) + ' <img>' + os.path.join(IMAGE_DIR, f"{file_name}-{page_num}-{i}.png") + '</img>'
    response = data['response'][i].strip()

    qwen_data.append({
        "id": f"identity_{i}",
        "conversations": [
            {
                "from": "user",
                "value": user_query
            },
            {
                "from": "assistant",
                "value": response
            }
        ]
    })

print(qwen_data[0])
print(len(qwen_data))

with open("ft-"+FILE_PATH, "w") as f:
    json.dump(qwen_data, f, indent=4)