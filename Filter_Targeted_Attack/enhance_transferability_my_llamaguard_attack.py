#!/usr/bin/env python
"""Local runner adapted from enhance_transferability_my_llamaguard_attack.py.

Preserved: Guard 1 gradient, first-step multi-position sampling (including the
original gather/view layout), subsequent single-position sampling, top-k 256,
joint Guard 1 + Guard 2 candidate loss, evaluator suffix length 20, and exact
safe/safe early stopping. Optimization uses the original Guard 1 prompt for
both models. Guard 2 classification uses Meta's official Guard 2 formatter.

Changes: CLI paths, frozen model parameters (input gradients remain enabled),
fresh per-forward DynamicCache objects, streamed candidate losses, incremental
result saves, and run metadata. These are not a claim of bitwise reproduction.
The replacement prompt dataset is not verified as the author's missing input.

Required sibling files: prompt_format.py and meta_guard_prompt_format.py.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run the MFA joint Guard 1/2 suffix demo locally."
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/LlamaGuard-7b",
        help="Local Llama Guard 1 directory.",
    )
    parser.add_argument(
        "--evaluator-model-path",
        default="/path/to/Meta-Llama-Guard-2-8B",
        help="Local Llama Guard 2 directory.",
    )
    parser.add_argument(
        "--prompt-file", "--toxic_prompt_path",
        dest="prompt_file",
        default=str(
            project_dir / "Evaluation_codes" / "attacked_prompts"
            / "40_raw_questions_extracted.pt"
        ),
    )
    parser.add_argument(
        "--output-file",
        default=str(
            project_dir / "outputs" / "suffix_raw_40" / "results.pt"
        ),
        help="New .pt output; existing files are rejected.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--iterations", "--num_iteration",
        dest="iterations", type=int, default=25,
    )
    parser.add_argument(
        "--candidate-size", "--candidate_size",
        dest="candidate_size", type=int, default=512,
    )
    parser.add_argument(
        "--candidate-batch-size", "--candidate_batch_size",
        dest="candidate_batch_size", type=int, default=32,
    )
    parser.add_argument(
        "--seed", "--random_seed",
        dest="seed", type=int, default=77,
    )

    args = parser.parse_args()

    for field in ("iterations", "candidate_size", "candidate_batch_size"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.candidate_size % args.candidate_batch_size:
        parser.error(
            "--candidate-size must be divisible by --candidate-batch-size."
        )

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(tokenizer, text):
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")


def fresh_cache(prefix, batch_size=1):
    # The immutable prefix tensors are never passed directly as a mutable cache.
    layers = tuple(
        (key.repeat(batch_size, 1, 1, 1), value.repeat(batch_size, 1, 1, 1))
        for key, value in prefix
    )
    return DynamicCache.from_legacy_cache(layers)


def prepare_context(model, tokenizer, prompt, before, after):
    embedding = model.get_input_embeddings()
    prompt_ids = encode(tokenizer, prompt)
    with torch.no_grad():
        context = torch.cat((embedding(encode(tokenizer, before)), embedding(prompt_ids)), dim=1)
        output = model(inputs_embeds=context, use_cache=True)
        cache = output.past_key_values
        legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        prefix = tuple((key.detach(), value.detach()) for key, value in legacy)
        target_ids = encode(tokenizer, "safe")
        after_embeddings = embedding(encode(tokenizer, after)).detach()
        target_embeddings = embedding(target_ids).detach()
    return {
        "prefix": prefix,
        "after": after_embeddings,
        "target": target_embeddings,
        "target_ids": target_ids,
        "prompt_ids": prompt_ids,
    }


def compute_loss(input_embeddings, target_embeddings, logits, target_ids):
    # Same shifted cross-entropy and FP16 conversion as the source script.
    offset = input_embeddings.shape[1] - target_embeddings.shape[1]
    shifted = logits[..., offset - 1:-1, :].contiguous()
    labels = target_ids.repeat(input_embeddings.shape[0], 1)
    losses = F.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]).to(torch.float16),
        labels.reshape(-1), reduction="none",
    )
    return losses.view(input_embeddings.shape[0], -1).mean(dim=1)


def assemble(suffix_embeddings, context):
    batch = suffix_embeddings.shape[0]
    return torch.cat((suffix_embeddings, context["after"].repeat(batch, 1, 1),
                      context["target"].repeat(batch, 1, 1)), dim=1)


def initial_sample_control(gradient, candidate_size):
    top_indices = (-gradient).topk(256, dim=1).indices
    positions = torch.randint(0, 256, (top_indices.shape[0], candidate_size), device=gradient.device)
    # Deliberately preserve the source gather/view arrangement; do not transpose.
    return top_indices.gather(1, positions).view(candidate_size, -1)


def sample_control(adv_ids, gradient, candidate_size):
    top_indices = (-gradient).topk(256, dim=1).indices
    candidates = adv_ids.repeat(candidate_size, 1)
    positions = torch.arange(0, len(adv_ids), len(adv_ids) / candidate_size,
                             device=gradient.device).to(torch.int64)
    # Keep the source CPU random draw followed by transfer to CUDA.
    samples = torch.randint(0, 256, (candidate_size, 1)).cuda()
    values = torch.gather(top_indices[positions], 1, samples)
    return candidates.scatter_(1, positions.unsqueeze(-1), values)


def candidate_losses(model, candidate_ids, context, batch_size):
    losses = []
    with torch.no_grad():
        for start in range(0, candidate_ids.shape[0], batch_size):
            ids = candidate_ids[start:start + batch_size].to("cuda")
            inputs = assemble(model.get_input_embeddings()(ids), context)
            output = model(inputs_embeds=inputs,
                           past_key_values=fresh_cache(context["prefix"], ids.shape[0]),
                           use_cache=True)
            losses.append(compute_loss(inputs, context["target"], output.logits,
                                       context["target_ids"]).detach())
            del output, inputs
    return torch.cat(losses)


def moderate(text, model, tokenizer, guard2=False):
    if guard2:
        body = build_default_prompt(
            agent_type=MetaAgentType.USER,
            conversations=[MetaConversationTurn(text, MetaAgentType.USER)],
            llama_guard_version=LlamaGuardVersion.LLAMA_GUARD_2,
        )
        # The official helper supplies [INST], but not the BOS token.
        if tokenizer.bos_token is None:
            raise ValueError("Guard 2 tokenizer has no BOS token.")
        input_ids = encode(tokenizer, tokenizer.bos_token + body)
    else:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], return_tensors="pt"
        ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0,
            return_dict_in_generate=True, output_scores=True,
        )
    first_probabilities = F.softmax(output.scores[0], dim=1)
    predicted_id = first_probabilities.argmax().item()
    probability = first_probabilities.max().item()
    # Preserve the source diagnostic heuristic; this is not a calibrated risk score.
    score = 1 - probability if predicted_id in (9109, 19193) else probability
    prediction = tokenizer.decode(output.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return prediction, score


def attack_one(prompt, model, tokenizer, evaluator, evaluator_tokenizer, args, before, after):
    context = prepare_context(model, tokenizer, prompt, before, after)
    # Intentionally preserve the source's shared optimization prompt format.
    evaluator_context = prepare_context(evaluator, evaluator_tokenizer, prompt, before, after)
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        vocab_embeddings = embedding(torch.arange(tokenizer.vocab_size, device="cuda")).detach()
    adv_ids = encode(tokenizer, "N q V w Y S V P H b D X p P d k h x E p")
    trace, prompts, suffixes, predictions = [], [], [], []
    evaluator_predictions = []
    for step in range(args.iterations):
        onehot = torch.zeros((1, adv_ids.shape[1], tokenizer.vocab_size),
                             dtype=model.dtype, device="cuda")
        onehot.scatter_(2, adv_ids.unsqueeze(2), 1.0).requires_grad_()
        suffix_embeddings = (onehot.squeeze(0) @ vocab_embeddings).unsqueeze(0)
        inputs = assemble(suffix_embeddings, context)
        output = model(inputs_embeds=inputs, past_key_values=fresh_cache(context["prefix"]), use_cache=True)
        loss = compute_loss(inputs, context["target"], output.logits, context["target_ids"])
        gradient = torch.autograd.grad(loss.sum(), onehot)[0][0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Non-finite gradient at step {step + 1}.")
        if step == 0:
            candidates = initial_sample_control(gradient, args.candidate_size)
        else:
            candidates = sample_control(adv_ids[0], gradient, args.candidate_size)
        del output, loss, gradient, inputs, suffix_embeddings, onehot
        decoded = tokenizer.batch_decode(candidates, skip_special_tokens=True)
        evaluator_ids = evaluator_tokenizer(
            decoded, return_tensors="pt", add_special_tokens=False,
            truncation=True, padding="max_length", max_length=20,
        )["input_ids"]
        losses1 = candidate_losses(model, candidates, context, args.candidate_batch_size)
        losses2 = candidate_losses(evaluator, evaluator_ids, evaluator_context, args.candidate_batch_size)
        total = losses1 + 1.0 * losses2
        total = torch.where(torch.isnan(total), torch.full_like(total, float("inf")), total)
        if not torch.isfinite(total).any():
            raise RuntimeError(f"All candidate losses are non-finite at step {step + 1}.")
        adv_ids = candidates[total.argmin()].unsqueeze(0)
        trace.append(total.min().item())
        suffixes.append(tokenizer.decode(adv_ids[0]))
        full_text = tokenizer.decode(torch.cat((context["prompt_ids"], adv_ids), dim=1)[0])
        prompts.append(full_text)
        prediction, score = moderate(full_text, model, tokenizer)
        evaluator_prediction, evaluator_score = moderate(full_text, evaluator, evaluator_tokenizer, guard2=True)
        predictions.append([prediction, score])
        evaluator_predictions.append([evaluator_prediction, evaluator_score])
        print(f"Step {step + 1}/{args.iterations} | Loss: {trace[-1]:.6f}", flush=True)
        print(f"Guard 1: {prediction!r} | Guard 2: {evaluator_prediction!r}", flush=True)
        if prediction == "safe" and evaluator_prediction == "safe":
            break
    return (trace, prompts, suffixes, predictions), evaluator_predictions


def atomic_save_results(results, output):
    temporary = output.with_name(output.name + ".tmp")
    torch.save(results, temporary)
    os.replace(temporary, output)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Import GPU libraries only after selecting visible devices.
    global torch, np, F, DynamicCache
    global build_default_prompt, MetaAgentType, MetaConversationTurn, LlamaGuardVersion
    import torch
    import numpy as np
    import torch.nn.functional as F
    from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, DynamicCache
    from prompt_format import build_prompt, AgentType, LLAMA_GUARD_CATEGORY, ConversationTurn
    from meta_guard_prompt_format import (
        build_default_prompt, AgentType as MetaAgentType,
        ConversationTurn as MetaConversationTurn, LlamaGuardVersion,
    )
    output = Path(args.output_file).expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.suffix != ".pt":
        raise ValueError("--output-file must end in .pt.")
    for path in (output, metadata_path, output.with_name(output.name + ".tmp")):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    for model_path in (args.model_path, args.evaluator_model_path):
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(f"Model configuration missing: {model_path}")
    prompts = torch.load(args.prompt_file, map_location="cpu", weights_only=True)
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("Expected a non-empty list or tuple of prompts.")
    if not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("All prompts must be non-empty strings.")
    prompts = prompts[:args.limit] if args.limit is not None else prompts
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, padding_side="left", local_files_only=True)
    evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.unk_token
    evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    for tok, expected in ((tokenizer, 9109), (evaluator_tokenizer, 19193)):
        if tok.encode("safe", add_special_tokens=False) != [expected]:
            raise ValueError("Safe token ID differs from the source script.")
    template = "[INST] " + build_prompt(
        AgentType.USER, LLAMA_GUARD_CATEGORY,
        [ConversationTurn("<user message>", AgentType.USER)],
    ) + " [/INST]"
    before, after = template.split("<user message>")
    print(f"Samples: {len(prompts)} | Output: {output}", flush=True)
    print("Optimization: source Guard 1 format for both models.", flush=True)
    print("Guard 2 classification: official Meta Guard 2 format.", flush=True)
    models = []
    for path in (args.model_path, args.evaluator_model_path):
        print(f"Loading: {path}", flush=True)
        models.append(LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="cuda", local_files_only=True,
        ).eval().requires_grad_(False))
    model, evaluator = models
    # Match the source: seed after loading model weights.
    set_seed(args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    import transformers
    source_hashes = {}
    for name in ("prompt_format.py", "meta_guard_prompt_format.py", Path(__file__).name):
        path = Path(__file__).resolve().parent / name
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "arguments": vars(args), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "source_sha256": source_hashes,
        "prompt_sha256": hashlib.sha256(Path(args.prompt_file).read_bytes()).hexdigest(),
        "optimization_template": "Original Guard 1 format shared by both models",
        "guard2_classification_template": "Meta official build_default_prompt, Guard 2",
        "input_provenance": "User-supplied prompts; equivalence to missing author input unverified",
        "completed_samples": 0, "evaluator_predictions": [],
    }
    with metadata_path.open("x", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=True, indent=2)
    # Reserve the result name without overwriting another run's output.
    with output.open("xb") as fout:
        torch.save([], fout)
    results = []
    for index, prompt in enumerate(prompts):
        print(f"\nSample {index + 1}/{len(prompts)}", flush=True)
        started = time.monotonic()
        result, evaluator_predictions = attack_one(
            prompt, model, tokenizer, evaluator, evaluator_tokenizer, args, before, after,
        )
        results.append(result)
        atomic_save_results(results, output)
        metadata["completed_samples"] = len(results)
        metadata["evaluator_predictions"].append(evaluator_predictions)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved {len(results)} samples | Elapsed: {time.monotonic() - started:.1f}s", flush=True)
        torch.cuda.empty_cache()
    print(f"Completed. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
