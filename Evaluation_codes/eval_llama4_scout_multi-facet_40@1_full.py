from together import Together
import os
import base64
from PIL import Image
import io
import torch
import json
import time

os.environ["http_proxy"] = "http://127.0.0.1:7891"
os.environ["https_proxy"] = "http://127.0.0.1:7891"

os.environ["TOGETHER_API_KEY"] = ""  # put your together api key here

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        import io
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return base64.b64encode(img_buffer.read()).decode('utf-8')

client = Together()

prompts_eval = []
prompt_file = "./attacked_prompts/transfer_5_length_good_start_llamaguard2_with_llamaguard3_attack_mannual_40.pt"
image_path = "./224_images/39_bad_prompt.bmp"

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

base64_image = encode_image(image_path)

MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MAX_RETRIES = 5

json_f = "./llama4_scout_multi-facet_40@1_full.jsonl"
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
                    # Get the response from the Together API call with image
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt_to_eval
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=4000,
                        temperature=0.7
                    )
                    round_data["response"] = response.choices[0].message.content
                    break
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in 20 seconds...")
                    time.sleep(20)

                    # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
            print(round_data["response"])
            print(f"prompt {i} round {round_number} finished.\n")
            time.sleep(10)
        print(prompt_data)
        # Add the prompt data to the results list
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")