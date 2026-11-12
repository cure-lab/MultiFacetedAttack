from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
from PIL import Image
import csv
import os

# replace prompt with the attacked ones
prompts_eval = []
# prompt_file = "./attacked_prompts/full_25_step_transfer_10_length_good_start_llamaguard2_with_llamaguard1_attack_40_mannual.pt"
# prompt_file = "./attacked_prompts/transfer_5_length_good_start_llamaguard2_with_llamaguard3_attack_mannual_40.pt"
prompt_file = "./attacked_prompts/25th_step_llamaguard1_good_start_mannual_40_clean.pt"


prompts_eval = torch.load(prompt_file)
print(prompts_eval)
print(len(prompts_eval))


adv_image = "./448_images/short_system_prompt_adv_images_1cos_0mse_new_448_6000/adversarial_image_test_448_w_clamp_w_norm1_hd.png"
img = Image.open(adv_image).convert("RGB")


import torch
from transformers import AutoTokenizer, AutoModel
import math
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 80
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "./NVLM-D-72B"
device_map = split_model()
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True,
    device_map=device_map).eval()

print(model)

image_len = len("Picture 1: <img></img>\n")

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
config = model.config

# Set pad_token_id to eos_token_id if not already set
if config.pad_token_id is None:
    config.pad_token_id = config.eos_token_id

# generation_config = dict(max_new_tokens=600, do_sample=True, top_p=None, top_k=None, pad_token_id=config.eos_token_id)
generation_config = dict(max_new_tokens=1000, do_sample=True, temperature=1.6, top_p=0.9 , top_k=50, tokpad_token_id=config.eos_token_id)

# generation_config = dict(max_new_tokens=1024)

json_f = "./multi-facet_results_40@10/NVLM_448.jsonl"

with open(json_f, "a", encoding="utf-8") as jsonl_file:
    # Iterate over each prompt
    for i, prompt_to_eval in enumerate(prompts_eval):
        pixel_values = load_image(adv_image, max_num=6).to(torch.bfloat16)
        prompt_data = {
            "prompt_id": i,
            "responses": []
        }
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_to_eval}
            ]}
        ]
        question = "<image>\n" + prompt_to_eval
        # question = "<image>\nPlease describe the image shortly."
        # Repeat each prompt (1 time in this case)
        # print(prompt_to_eval)
        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            # print(response)
            round_data["response"] = response
            print(response)
            # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
        print("=====================================")
        print("=====================================")
        print("=====================================")
        # print("prompt_data: ", prompt_data)
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")