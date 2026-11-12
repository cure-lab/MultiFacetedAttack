#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "Mutifaceted_atk_logs_jsonl"

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local MiMo-VL-7B-SFT using the original MFA attack samples."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/MiMo-VL-7B-SFT",
        help="Path to the local MiMo-VL-7B-SFT model directory.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(
            script_dir
            / "attacked_prompts"
            / (
                "transfer_5_length_good_start_llamaguard2_"
                "with_llamaguard3_attack_mannual_40.pt"
            )
        ),
        help="Path to the pre-generated MFA attack prompts.",
    )
    parser.add_argument(
        "--image-path",
        default=str(script_dir / "224_images" / "39_bad_prompt.png"),
        help="Path to the adversarial image.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(
            results_dir
            / (
                "mimo_vl_7b_sft_image_attack_40@1_224.jsonl"
            )
        ),
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU identifier to expose through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=12,
        help="Index of the saved attack prompt selected for each sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples; defaults to all 40 samples.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum combined input and output context length; "
            "defaults to the model configuration."
        ),
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0.")
    if args.prompt_index < 0:
        parser.error("--prompt-index must be non-negative.")
    return args


def main():
    args = parse_args()

    # Set GPU visibility before importing torch or vLLM.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from PIL import Image
    from vllm import LLM, SamplingParams

    model_path = Path(args.model_path).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    image_path = Path(args.image_path).resolve()
    output_file = Path(args.output_jsonl).resolve()

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {model_path / 'config.json'}"
        )

    for path in (prompt_file, image_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    if output_file.exists():
        raise FileExistsError(
            f"Output file already exists; refusing to overwrite: {output_file}\n"
            "Specify a new filename with --output-jsonl."
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Default to the 13th saved attack prompt, matching the original script.
    prompts_40 = torch.load(
        prompt_file,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(prompts_40, (list, tuple)) or len(prompts_40) != 40:
        raise ValueError(
            "The attack file must contain 40 records. Check --prompt-file."
        )

    prompts_eval = []
    for i, record in enumerate(prompts_40):
        if not isinstance(record, (list, tuple)) or len(record) != 4:
            raise ValueError(f"Unexpected record structure at index {i}.")

        candidates = record[1]
        if not isinstance(candidates, (list, tuple)):
            raise TypeError(
                f"The prompt collection at record {i} must be a list or tuple."
            )
        if args.prompt_index >= len(candidates):
            raise IndexError(
                f"Record {i} contains only {len(candidates)} prompts; "
                f"prompt index {args.prompt_index} is out of range."
            )

        prompt = candidates[args.prompt_index]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"The selected prompt at record {i} must be a non-empty string."
            )
        prompts_eval.append(prompt)

    if args.limit is not None:
        prompts_eval = prompts_eval[:args.limit]

    # Convert to RGB, then pass a lossless PNG to vLLM.
    with Image.open(image_path) as image:
        image_pil = image.convert("RGB")

    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_url = f"data:image/png;base64,{encoded_image}"

    print(f"Model: {model_path}", flush=True)
    print(f"Attack prompts: {prompt_file}", flush=True)
    print(f"Prompt index: {args.prompt_index}", flush=True)
    print(f"Image: {image_path}, size: {image_pil.size}", flush=True)
    print(f"Samples to evaluate: {len(prompts_eval)}", flush=True)
    print(f"Output: {output_file}", flush=True)

    llm_kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "disable_log_stats": False,
        "limit_mm_per_prompt": {"image": 1},
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    # Default sampling parameters match the original script.
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Exclusive creation prevents overwriting existing results.
    with output_file.open("x", encoding="utf-8") as fout:
        for i, prompt_to_eval in enumerate(prompts_eval):
            conv = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_to_eval,
                    },
                ],
            }]

            # Propagate errors immediately; completed records remain on disk.
            outs = llm.chat(
                conv,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            completion = outs[0].outputs[0]
            gen = completion.text.strip()

            # Preserve the original output fields for evaluator compatibility.
            record = {
                "prompt_id": i,
                "prompt": prompt_to_eval,
                "image": str(image_path),
                "responses": [
                    {"round": 1, "response": gen}
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{i + 1}/{len(prompts_eval)}] Completed, "
                f"generated {len(completion.token_ids)} tokens, "
                f"finish reason: {completion.finish_reason}",
                flush=True,
            )

    print(f"\nResponses saved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()