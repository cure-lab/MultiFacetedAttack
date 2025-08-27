# MultiFaceted Attack: Breaking GPT-OSS with Reward Hacking and Reasoning Breaks

This repository is the offical implementation of our kaggle openai-gpt-oss-20b-red-teaming challange submission **MultiFaceted Attack: Breaking GPT-OSS with Reward Hacking and Reasoning Breaks**. 

## Table of Contents

- [Installation](#installation)
- [Experiments](#experiments)
- [Citation](#citation)

## Installation

We need the newest version of FastChat `fschat==0.2.23` and please make sure to install this version. The `llm-attacks` package can be installed by running the following command at the root of this repository:

```bash
pip install -e .
```

## Experiments 

The `experiments` folder contains code to reproduce GCG experiments on AdvBench.

- To run attacks on one target prompts and all target prompts  

```bash
python ./demo_gcg_gptoss.py
python ./demo_gcg_batch_attack_all_new_prompts.py
```

- To perform evaluation experiments, please refer to `demo_gptoss_evaluation.ipynb`.

## Citation
If you find this useful in your research, please consider citing:

```
@misc{yang2025effectiveblackboxmultifacetedattacks,
      title={Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails}, 
      author={Yijun Yang and Lichao Wang and Xiao Yang and Lanqing Hong and Jun Zhu},
      year={2025},
      eprint={2502.05772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.05772}, 
}
```

## Note 
This paper is derived the `llm-attacks` repository: https://github.com/llm-attacks/llm-attacks