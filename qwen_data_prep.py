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

import argparse
import json
import os
import sys
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from base64 import b64decode
import random

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Qwen fine-tuning")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("--image_dir", default="images", help="Directory to save images (default: images)")
    args = parser.parse_args()
    
    FILE_PATH = args.input_file
    IMAGE_DIR = args.image_dir

    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # load the dataset we have
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    print(data[0].keys()) # dict_keys(['time', 'file_name', 'page_num', 'table_num', 'image', 'response'])

    randomized_user_prompt = [
    """
**Role**
Convert every piece of structured text in a scanned image into accurate HTML tables.

### Workflow

1. **Analyze Image**

   * Find *all* structured info: tables, headers, key‑value pairs, lists, metadata, implied columns/rows.
   * Note spans (row/col), poor quality areas, and extract **only English text**.

2. **Map Structure**

   * Everything becomes a table.

     * Headers/titles → 1 row × 1 col table.
     * Key–value → 2‑column table.
     * Preserve actual rows, cols, rowspans, colspans.
   * Include company names, titles, model/make, part/drawing numbers, section headers, etc.

3. **Draft Extraction**

   * Build raw HTML: `<table><thead>…</thead><tbody>…</tbody></table>` for each unit.
   * Keep empty cells, correct spans.
   * Replace logos with `logo here`, images with `image here`.

4. **Verify & Refine**

   * Character‑by‑character check: no omissions, misreads, or added text.
   * Ensure span accuracy, no unintended merges, no non‑English.
   * Reverse descending tables to ascending order but keep “No.” values intact.

5. **Finalize**

   * Output *all* tables, wrapped once in `<final> … </final>`.
   * Provide a brief summary after the code block.

### Strict Rules

* **Everything** visible becomes a table—no exceptions.
* HTML only: no classes/ids/styles, no nested tables, no `<caption>`.
* Use newline escape (`&#10;`) instead of `<br>` inside cells.
* One `<final>` wrapper only.
* No hallucinations, merges, or missing content; no extra information.
"""
    ]

    qwen_data = []

    data_idx = 0

    for i in tqdm(range(len(data))):

        time = data[i]['time']
        file_name = data[i]['file_name']
        page_num = data[i]['page_num']

        # save the image wit unique name
        base64_image = data[i]['image']
        image = Image.open(BytesIO(b64decode(base64_image)))
        image.save(os.path.join(IMAGE_DIR, f"{file_name}-{page_num}-{i}.png"))

        user_query = random.choice(randomized_user_prompt) + ' <img>' + os.path.join(IMAGE_DIR, f"{file_name}-{page_num}-{i}.png") + '</img>'
        response = data[i]['response'].strip()
        
        if '<img' not in response:
          qwen_data.append({
              "id": f"identity_{data_idx}",
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
          data_idx += 1

    print(qwen_data[0])
    print(len(qwen_data))

    output_file = f"ft-{os.path.basename(FILE_PATH)}"
    with open(output_file, "w") as f:
        json.dump(qwen_data, f, indent=4)

if __name__ == "__main__":
    main()