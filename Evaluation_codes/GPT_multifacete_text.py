from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
from PIL import Image
import csv
import os


prompts_eval = []
prompt_file = "./attacked_prompts/transfer_5_length_good_start_llamaguard2_with_llamaguard3_attack_mannual_40.pt"

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
    max_score_idx = 12

    prompts_eval.append(prompt_strs[max_score_idx])

print(prompts_eval)
print(len(prompts_eval))


from poe_api_wrapper import AsyncPoeApi
import asyncio
from poe_api_wrapper import PoeApi
import time
import os

# Set the HTTP and HTTPS proxy environment variables
os.environ["HTTP_PROXY"] = "http://localhost:7890"
os.environ["HTTPS_PROXY"] = "http://localhost:7890"

# update refering to poe-api
tokens = {
    'p-b': '',
    'p-lat': '', # new
}

bot = "gpt4_o"
client = PoeApi(tokens=tokens)  

json_f = "./GPT-4o@1_demo.jsonl"
with open(json_f, "a", encoding="utf-8") as jsonl_file:
    # Iterate over each prompt
    for i, prompt_to_eval in enumerate(prompts_eval):
        # print(image_path)
        prompt_data = {
            "prompt_id": i,
            "responses": []
        }
        # Repeat each prompt (1 time in this case)
        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }
            while True:
                try:
                    # Get the response from the send_message call
                    for chunk in client.send_message(bot, prompt_to_eval):
                        round_data["response"] += chunk["response"]
                    break
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in 20 seconds...")
                    time.sleep(20)

                    # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
            print( round_data["response"])
            print(f"prompt {i} round {round_number} finished.\n")
            time.sleep(10)
        print(prompt_data)
        # Add the prompt data to the results list
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")