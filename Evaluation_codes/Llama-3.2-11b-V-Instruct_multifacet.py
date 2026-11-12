
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import csv
import os


# replace prompt with the attacked ones
prompts_eval = []

prompt_file = "./attacked_prompts/25th_step_llamaguard1_good_start_mannual_40_clean.pt"

prompts_40 = torch.load(prompt_file)
prompts_eval = prompts_40
print(prompts_eval)
print(len(prompts_eval))

# figstep_image = "./39_bad_prompt.bmp"
figstep_image = "./adversarial_image_1_test_448_w_clamp_w_norm_work.png"
figstep_image = Image.open(figstep_image).convert('RGB')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = ""  # put your model path here

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

import json
import time

json_f = "./Llama-3.2-11B-Vision-Instruct@1_clean.jsonl"
# Open the jsonl file for writing
with open(json_f, "a", encoding="utf-8") as jsonl_file:
    # Iterate over each prompt
    for i, prompt_to_eval in enumerate(prompts_eval):
        prompt_data = {
            "img_id": i,
            "responses": []
        }
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_to_eval}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            figstep_image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }
            output = model.generate(**inputs, max_new_tokens=1000, temperature=1.2)
            response = processor.decode(output[0])[len(input_text):-10]
            print(response)
            round_data["response"] = response
            # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
        # print("prompt_data: ", prompt_data)
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")
