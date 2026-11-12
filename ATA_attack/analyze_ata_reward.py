#!/usr/bin/env python3
"""Score paired ATA responses with a scalar Hugging Face reward model."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def read_pairs(path):
    rows, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("prompt_id")
            if type(pid) is not int or pid in seen:
                raise ValueError(f"Invalid or duplicate prompt_id at line {line_no}.")
            seen.add(pid)
            for key in ("ata_prompt", "baseline_response", "ata_response"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing/empty {key} at line {line_no}.")
            rows.append(row)
    if not rows:
        raise ValueError("Input contains no pairs.")
    return rows


def summarize(rows):
    deltas = [r["delta"] for r in rows]
    n = len(rows)
    return {
        "num_pairs": n,
        "baseline_mean_reward": mean(r["baseline_score"] for r in rows),
        "ata_mean_reward": mean(r["ata_score"] for r in rows),
        "mean_delta_reward": mean(deltas),
        "wins": sum(d > 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
        "losses": sum(d < 0 for d in deltas),
        "win_rate_percent": 100 * sum(d > 0 for d in deltas) / n,
        "generation_length_limited_pairs": sum(r["generation_length_limited"] for r in rows),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local scalar reward model directory.")
    parser.add_argument("--input-jsonl", default=str(root / "outputs/ata/mimo_ata_smoke.jsonl"))
    parser.add_argument("--output-jsonl", required=True, help="Per-pair scores; summary uses .summary.json.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Reject longer inputs; never silently truncate. Defaults to model limits.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    for key in ("limit", "max_length"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"{key} must be positive.")
    source = Path(args.input_jsonl).expanduser().resolve()
    output = Path(args.output_jsonl).expanduser().resolve()
    summary_path = output.with_suffix(".summary.json")
    for path in (output, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    pairs = read_pairs(source)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    model_path = Path(args.model_path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("Reward tokenizer has no chat template; supply the correct checkpoint/tokenizer.")
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), torch_dtype=getattr(torch, args.dtype),
        local_files_only=True, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Checkpoint is incomplete for reward scoring: {loading}")
    if model.config.num_labels != 1:
        raise ValueError("Expected one scalar reward logit. This checkpoint needs a dedicated adapter.")
    model.to(args.device).eval()
    limits = [v for v in (getattr(model.config, "max_position_embeddings", None),
                         tokenizer.model_max_length, args.max_length)
              if isinstance(v, int) and 0 < v < 1000000]
    if not limits:
        raise ValueError("Cannot determine context limit; specify --max-length.")
    context_limit = min(limits)

    def score(prompt, response):
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        # Tokenize the complete chat once, avoiding duplicate special tokens.
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True,
        )
        length = inputs["input_ids"].shape[-1]
        if length > context_limit:
            raise ValueError(f"Scoring input has {length} tokens, limit {context_limit}; no truncation applied.")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        if logits.numel() != 1:
            raise ValueError(f"Expected scalar reward, got shape {tuple(logits.shape)}.")
        value = logits.float().item()
        if not math.isfinite(value):
            raise ValueError("Non-finite reward score.")
        return value, length

    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("x", encoding="utf-8") as fout:
        for pair in pairs:
            baseline, n_base = score(pair["ata_prompt"], pair["baseline_response"])
            ata, n_ata = score(pair["ata_prompt"], pair["ata_response"])
            row = {
                "prompt_id": pair["prompt_id"], "baseline_score": baseline,
                "ata_score": ata, "delta": ata - baseline, "win": ata > baseline,
                "baseline_scoring_tokens": n_base, "ata_scoring_tokens": n_ata,
                "baseline_generation": pair.get("baseline_generation", {}),
                "ata_generation": pair.get("ata_generation", {}),
                "generation_length_limited": any(
                    pair.get(key, {}).get("finish_reason") == "length"
                    for key in ("baseline_generation", "ata_generation")),
            }
            fout.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            fout.flush()
            results.append(row)
            print(f"[{len(results)}/{len(pairs)}] baseline={baseline:.6f}, ATA={ata:.6f}, delta={ata-baseline:.6f}", flush=True)
    summary = summarize(results)
    summary.update(input_jsonl=str(source), reward_model=str(model_path), dtype=args.dtype,
                   scoring_prompt="ata_prompt for both responses", response_processing="verbatim",
                   context_limit=context_limit, input_truncation=False)
    with summary_path.open("x", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Scores saved to: {output}\nSummary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
