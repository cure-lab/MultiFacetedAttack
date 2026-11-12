from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import csv
import json
from PIL import Image
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out
    
model_path = "./llava-v1.5"

prompts_eval = []
# prompt_file = "./attacked_prompts/25th_step_llamaguard1_good_start_mannual_40_clean.pt"
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
    # max_score_idx = 18

    prompts_eval.append(prompts)

print(prompts_eval)
print(len(prompts_eval))
# for file in files:
#     print(file)
#     if "png" not in file:
#         continue
#     img = Image.open(os.path.join(files_dir, file))
#     # img.show()
#     figstep_images.append(img)
# print(len(figstep_images))

# figstep_images[0].show()

image_path = "./224_images/39_bad_prompt.bmp"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0.7,
    "top_p": 0.9,
    "num_beams": 1,
    "max_new_tokens": 512
})()

# eval_model(args)
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name)

with open("responses_llava_v15_multifacet_clean@1.jsonl", "a", encoding="utf-8") as jsonl_file:
    for i, prompt_to_eval in enumerate(prompts_eval):
        prompt_data = {
            "prompt_id": i,
            "prompt": prompt_to_eval,
            "responses": []
        }
        qs = prompt_to_eval
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = load_images([image_path])
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)
            round_data["response"] = outputs
            # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
        print("prompt_data: ", prompt_data)
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")