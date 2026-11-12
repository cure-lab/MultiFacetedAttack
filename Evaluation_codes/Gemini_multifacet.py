import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

import csv
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
FinishReason = gapic_content_types.Candidate.FinishReason


import json
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, GenerationConfig
from vertexai.generative_models import (
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    # HarmBlockMethod,    
    Part,
)

import vertexai
from io import BytesIO
import base64

image_path = "./448_images/adversarial_image_1_test_448_w_clamp_w_norm_work.png"

image = Image.open(image_path)

# Convert the image to raw bytes
buffer = BytesIO()
image.save(buffer, format="PNG")  # Adjust format if using PNG, etc.
# image.save(buffer, format="BMP")  # Adjust format if using PNG, etc.

image_data = buffer.getvalue()

# Create a Part object with the raw image data and MIME type
image_part = Part.from_data(data=image_data, mime_type="image/png")  # Adjust MIME type if needed

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# TODO: Update ID
PROJECT_ID = " " 
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-exp-1206")  # replace here for different gemini models


# replace prompt with the attacked ones
prompts_eval = []
prompt_file = "./attacked_prompts/full_25_step_transfer_10_length_good_start_llamaguard2_with_llamaguard1_attack_40_mannual.pt"
prompts_40 = torch.load(prompt_file)
print(len(prompts_40)) # 40
for prompts in prompts_40:  # out of 4
    prompt_strs = prompts[1]
    prompt_scores = prompts[3]
    # print(prompt_scores)
    scores = []
    for prompt_score in prompt_scores:
        score = prompt_score[1]
        scores.append(score)
    max_score_idx = 16
    # max_score_idx = 12
    # max_score_idx = 18
    prompts_eval.append(prompt_strs[max_score_idx])

print(prompts_eval)
print(len(prompts_eval))

import time

json_f = "responses_gemini-v2.0-pro_multifacet@1.jsonl"


with open(json_f, "a", encoding="utf-8") as jsonl_file:
    # Iterate over each prompt
    to_moderation = 0
    for i, prompt_to_eval in enumerate(prompts_eval):  # 12
        # if i<= 23:
        #     continue
        prompt_data = {
            "prompt_id": i,
            "prompt": prompt_to_eval,
            "responses": []
        }
        mode_flag = 0
        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }
            response = model.generate_content(
                [
                    image_part,
                    prompt_to_eval,
                ]
            )
            print("response.candidates[0].finish_reason: ", response.candidates[0].finish_reason)
            # print("response.candidates[0].finish_reason: ", response.candidates[0].finish_reason.number)
            if response.candidates[0].finish_reason == FinishReason.SAFETY:
                print("unsafe")
                round_data["response"] = "Rejected by model, FinishReason.SAFETY"
                prompt_data["responses"].append(round_data)
                mode_flag = 1
            else:
                print(response.text)
                round_data["response"] = response.text
                prompt_data["responses"].append(round_data)
            time.sleep(30)
        
        if mode_flag == 1:
            to_moderation += 1

        # print("prompt_data: ", prompt_data)
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

        print(f"success to moderation: {to_moderation}")