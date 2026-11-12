#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract InternVL-Chat-V1-5 modules for MFA."
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/InternVL-Chat-V1-5",  # Set your local model path.
    )

    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )

    parser.add_argument(
        "--gpu",
        default="0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU visibility before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from transformers import AutoModel

    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {model_path / 'config.json'}"
        )

    filenames = (
        "vision_encoder.pth",
        "mlp1.pt",
        "internvl_tok_embeddings.pt",
    )
    for filename in filenames:
        output_path = output_dir / filename
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"GPU: {args.gpu}", flush=True)

    # Load the same full model used by the original extraction script.
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()

    # Preserve the original module selection and serialization formats.
    vision_encoder = model.vision_model
    projector = model.mlp1
    embeddings_layer = model.language_model.model.tok_embeddings

    # Save CPU tensors so loading is independent of the original GPU.
    vision_state = {
        name: tensor.detach().cpu()
        for name, tensor in vision_encoder.state_dict().items()
    }
    output_path = output_dir / "vision_encoder.pth"
    torch.save(vision_state, output_path)
    del vision_state
    print(f"Saved: {output_path}", flush=True)

    output_path = output_dir / "mlp1.pt"
    torch.save(projector.cpu(), output_path)
    print(f"Saved: {output_path}", flush=True)

    output_path = output_dir / "internvl_tok_embeddings.pt"
    torch.save(embeddings_layer.cpu(), output_path)
    print(f"Saved: {output_path}", flush=True)

    print("Module extraction completed.", flush=True)


if __name__ == "__main__":
    main()