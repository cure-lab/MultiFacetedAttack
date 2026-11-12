# Multi-Faceted Attack on Vision-Language Models

This repository provides demonstration code for multi-faceted attacks on Vision-Language Models (VLMs), including evaluation scripts, pre-generated attack samples, attack logs, and source code for vision encoder and filter-targeted attacks.


## Environment Setup

Use Python 3.10 and install the dependencies:

```bash
python3.10 -m venv mfa
source mfa/bin/activate
python -m pip install -r requirements.txt
```

Run the following commands from the repository root. For open-source models, download the Hugging Face model weights and replace the `/path/to/...` defaults in the corresponding Python scripts with your local model paths. For commercial models, configure Vertex AI authentication and the required credentials for Together AI, xAI, or Poe.

## 📊 Evaluation

We provide pre-generated adversarial images (at 224px and 448px) and attacked prompts for evaluation purposes.

### Included Assets

```text
Evaluation_codes/
├── 224_images/
│   ├── 39_bad_prompt.bmp
│   └── 39_bad_prompt.png
├── 448_images/
│   └── adversarial_image_1_test_448_w_clamp_w_norm_work.png
└── attacked_prompts/
    ├── 25th_step_llamaguard1_good_start_mannual_40_clean.pt
    ├── full_25_steps_30_length_good_start_llamaguard3_attack_mannual_40_prompts.pt
    ├── transfer_5_length_good_start_llamaguard2_with_llamaguard3_attack_mannual_40.pt
    └── 40_raw_questions_extracted.pt
```

### Example: MiMo-VL Evaluation

Download `XiaomiMiMo/MiMo-VL-7B-SFT`, then run:

```bash
python -u Evaluation_codes/eval_mimo_multifact.py 
```
Responses are saved in JSONL format, with one record per sample:

```json
{"prompt_id": 0, "prompt": "...", "image": "...", "responses": [{"round": 1, "response": "..."}]}
```

### Response Safety Evaluation

Download Llama Guard 3 and evaluate the generated responses:

```bash
python -u Evaluation_codes/automated_evaluators/analyze_llamaguard_res.py 
```

Results are saved alongside the response log as:

```text
mimo_vl_7b_sft_image_attack_40@1_224_llamaguard_results.json
```

### Other Models

The evaluation scripts for other models are retained in `Evaluation_codes/`. Configure the corresponding API credentials or local model paths before running them. The local execution example above covers MiMo-VL.

## 📜 Attack Logs

We have released detailed logs from our attacks on various VLMs. The logs are stored in the `./Mutifaceted_atk_logs_jsonl` directory in `.jsonl` format for easy parsing and analysis.
 
### Available Logs
```bash
├── gemini_2_0_multifacet.jsonl
├── gemini-2.5-flash_multifacet.jsonl
├── GPT-4_1_mini_multifacet.jsonl
├── GPT-4_1_multifacet.jsonl
├── gpt-4o_multifacet_clean.jsonl
├── gpt4v_multifacet.jsonl
├── grok2_vision_multifacet.jsonl
├── Llama-3.2-11B-Vision-Instruct_multifacet.jsonl
├── llama4_scout_multifacet.jsonl
├── MiMo-VL-7B_multifacet.jsonl
├── minigpt4_multifacet.jsonl
├── mistral_large_multifacet.jsonl
├── NVLM_multifacet.jsonl
├── palm_multifacet.jsonl
├── qwen_vl_multifacet.jsonl
├── Rokecore_multifacet.jsonl
└── solar_mini_multifacet.jsonl
```


## ⚔️ Vision Encoder Attack

This section provides code for attacking the vision encoder of `InternVL-Chat-V1-5`, with support for multiple resolutions.

### 1. Prepare Model Modules

Download `OpenGVLab/InternVL-Chat-V1-5`, then extract the required modules:

```bash
python -u Vision_encoder_Internvl/save_intern_vl_modules.py 
```

This generates:

- `vision_encoder.pth`
- `mlp1.pt`
- `internvl_tok_embeddings.pt`


### 2. Prepare Images

Place the source images in:

```text
images_for_mulitfaceted_attack/
```

### 3. Run Attack

```bash
python -u Vision_encoder_Internvl/visual_attack_internvl_v1_5_support_multi_resolution.py 
```

Generated images are saved under `outputs/vision_attack/`.

## Filter-Targeted Attack

### 1. Prepare Models and Questions

Download Llama Guard 1 and Llama Guard 2.

The clean questions are stored in:

```text
Evaluation_codes/attacked_prompts/40_raw_questions_extracted.pt
```

### 2. Run Attack

```bash
python -u Filter_Targeted_Attack/enhance_transferability_my_llamaguard_attack.py
```

Results are saved after each completed sample to `outputs/suffix_raw_40/results.pt`. 



## Citation
If you find this useful in your research, please consider citing:

```
@misc{yang2025effectiveblackboxmultifacetedattacks,
      title={Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models}, 
      author={Yijun Yang* and Lichao Wang* and Jianping Zhang and Chi Harold Liu and Lanqing Hong and Qiang Xu},
      year={2026},
      archivePrefix={AAAI},
      primaryClass={cs.CV},
}
```
