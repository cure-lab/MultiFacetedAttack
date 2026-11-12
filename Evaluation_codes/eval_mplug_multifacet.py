import torch
from PIL import Image
from transformers import TextStreamer
import csv
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import json

model_path = './mplug-owl2-llama2-7b/mplug-owl2-llama2-7b'

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
    # max_score_idx = 18
    print(prompt_strs[max_score_idx])
    prompts_eval.append(prompt_strs[max_score_idx])

print(prompts_eval)
print(len(prompts_eval))

image_path = "./224_images/39_bad_prompt.bmp"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")


with open("responses_mplug-owl2_multi-facet_@1.jsonl", "a", encoding="utf-8") as jsonl_file:
    for i, prompt_to_eval in enumerate(prompts_eval):
        # if i > 20:
        #     continue
        prompt_data = {
            "prompt_id": i,
            "prompt": prompt_to_eval,
            "responses": []
        }
        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + prompt_to_eval
        inp = prompt_to_eval
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        for round_number in range(1, 2):
            round_data = {
                "round": round_number,
                "response": ""
            }

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=None,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            # print(output_ids.shape)
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:-1]).strip()
            # print(outputs)
            # quit()
            round_data["response"] = outputs
            # Add round data to the prompt's response list
            prompt_data["responses"].append(round_data)
        # print("prompt_data: ", prompt_data)
        # Write the prompt data to the jsonl file as a single line
        try:
            jsonl_file.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to file: {e}")
