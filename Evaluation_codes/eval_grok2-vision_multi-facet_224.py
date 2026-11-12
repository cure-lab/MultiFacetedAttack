#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import base64
import torch
from PIL import Image
from openai import OpenAI

os.environ["HTTP_PROXY"]  = "http://localhost:7891"
os.environ["HTTPS_PROXY"] = "http://localhost:7891"

PROMPT_FILE   = "./attacked_prompts/transfer_5_length_good_start_llamaguard2_with_llamaguard3_attack_mannual_40.pt"
IMAGE_PATH    = "./224_images/39_bad_prompt.bmp"
OUTPUT_JSONL  = "./grok2_vision_attack_results_224.jsonl"

MODEL_NAME    = "grok-2-vision-latest"
XAI_API_KEY   = "" # put your xai api key here
BASE_URL      = "https://api.x.ai/v1"
TEMPERATURE   = 0.7
MAX_RETRIES   = 5
SLEEP_BETWEEN = 10  # seconds between calls to respect rate-limit

prompts = torch.load(PROMPT_FILE)           # list of length 40
prompts_eval = []
for p in prompts:
    prompt_variants  = p[1]   # list[str]
    prompt_scores    = p[3]   # list[(id, score)]
    best_idx         = 12  
    prompts_eval.append(prompt_variants[best_idx])

print(f"Loaded {len(prompts_eval)} prompts.")

def encode_image_to_png_b64(path: str) -> str:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

base64_img = encode_image_to_png_b64(IMAGE_PATH)

client = OpenAI(api_key=XAI_API_KEY, base_url=BASE_URL)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for idx, adv_prompt in enumerate(prompts_eval, 1):
        prompt_record = {"prompt_id": idx, "prompt": adv_prompt, "responses": []}

        for round_no in range(1, 2):
            attempt = 0
            while attempt < MAX_RETRIES:
                try:
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=TEMPERATURE,
                        max_tokens=4000,
                        messages=[
                            {   
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_img}",
                                            "detail": "high"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": adv_prompt
                                    }
                                ]
                            }
                        ]
                    )

                    reply = res.choices[0].message.content
                    print(f"[Prompt {idx} • Round {round_no}]")
                    print(reply, "\n")

                    prompt_record["responses"].append({
                        "round": round_no,
                        "response": reply
                    })
                    break  

                except Exception as e:
                    attempt += 1
                    if attempt >= MAX_RETRIES:
                        print(f"‼️  Prompt {idx} round {round_no} failed after {MAX_RETRIES} retries.")
                        prompt_record["responses"].append({
                            "round": round_no,
                            "response": f"[ERROR] {e}"
                        })
                    else:
                        print(f"Error: {e} – retry {attempt}/{MAX_RETRIES} in 20 s.")
                        time.sleep(20)

            time.sleep(SLEEP_BETWEEN)  

        # Write one JSONL record.
        fout.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
        fout.flush()

print(f"\n✅  Evaluation finished. Results saved to {OUTPUT_JSONL}")
