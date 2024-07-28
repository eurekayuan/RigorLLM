# RigorLLM
This repo contains code for "RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content".

## Requirements
```
conda create -n rigorllm python=3.11
conda activate rigorllm
pip install -r requirements.txt
```

## Data Setup
All the datasets that we use are publicly available. 

Multi-turn Helpfulness: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl 

Download and place all the training and testing datasets under the `data` folder:
```
data
|-- advbench
|-- hexphi
|-- hotpot
|-- multiturn
    |-- question.jsonl
|-- openai
    |-- samples-1680.jsonl
```

## Energy-based Data Generation
`data_gen.py` generates augmented data based on existing datasets. The generated data will then be used for the KNN module. Our implementation is based on [COLD](https://github.com/qkaren/COLD_decoding). Optionally, further filter generated data by perplexity or toxicity to ensure higher data quality.

## Prompt Augmentation
`prompt_augmentation.py` generates prompt augmentations for test data using GPT.

## Content Moderation
First, generate LLM predictions with `save_llm_prediction.py`. The script will save LLM predictions under the `cache` folder. Then, run `moderation.py` to test content moderation performance.

To test resilience to jailbreaking attacks, please refer to [GCG](https://github.com/llm-attacks/llm-attacks) and [RPO](https://github.com/lapisrocks/rpo) for universal adversarial and safe suffix generation. After generating those suffices, simply append them during testing time.

