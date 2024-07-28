import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from moderation import preprocess_openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion, AgentType
from data_processing import *



def llamaguard_predict(model, tokenizer, original_prompt, agent_type=AgentType.USER):
    formatted_prompt = build_default_prompt(agent_type, create_conversation([original_prompt]), LlamaGuardVersion.LLAMA_GUARD_1)
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=0, return_dict_in_generate=True, output_scores=True)
    outputs_sequence, outputs_scores = outputs.sequences, outputs.scores
    
    results = tokenizer.decode(outputs_sequence[0][prompt_len:], skip_special_tokens=True)
    results = results.strip().split('\n')[0]  # disgard the second row which is the category
    probs_first_token = torch.softmax(outputs_scores[0][0], dim=-1)
    safe_token_id = int(tokenizer.encode("safe", return_tensors="pt").to("cuda")[0, 1])
    unsafe_token_id = int(tokenizer.encode("unsafe", return_tensors="pt").to("cuda")[0, 1])
    prob_safe = float(probs_first_token[safe_token_id])
    prob_unsafe = float(probs_first_token[unsafe_token_id])
    
    return results, prob_safe, prob_unsafe


def inference(args):
    if args.aug:
        querys, querys_category = preprocess_openai_plus()
        if os.path.exists('cache/llamaguard_probs_safe_plus.npy') and os.path.exists('cache/llamaguard_probs_unsafe_plus.npy'):
            probs_safe = np.load('cache/llamaguard_probs_safe_plus.npy')
            probs_unsafe = np.load('cache/llamaguard_probs_unsafe_plus.npy')
        else:
            model_id = "meta-llama/LlamaGuard-7b"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            probs_safe = []
            probs_unsafe = []
            for i, query in enumerate(tqdm(querys)):
                results, prob_safe, prob_unsafe = llamaguard_predict(model, tokenizer, query)
                probs_safe.append(prob_safe)
                probs_unsafe.append(prob_unsafe)
            probs_safe = np.array(probs_safe)
            probs_unsafe = np.array(probs_unsafe)
            np.save('cache/llamaguard_probs_safe_plus.npy', probs_safe)
            np.save('cache/llamaguard_probs_unsafe_plus.npy', probs_unsafe)
    else:
        querys, querys_category = preprocess_openai()
        if os.path.exists('cache/llamaguard_probs_safe.npy') and os.path.exists('cache/llamaguard_probs_unsafe.npy'):
            probs_safe = np.load('cache/llamaguard_probs_safe.npy')
            probs_unsafe = np.load('cache/llamaguard_probs_unsafe.npy')
        else:
            model_id = "meta-llama/LlamaGuard-7b"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            probs_safe = []
            probs_unsafe = []
            for i, query in enumerate(tqdm(querys)):
                results, prob_safe, prob_unsafe = llamaguard_predict(model, tokenizer, query)
                probs_safe.append(prob_safe)
                probs_unsafe.append(prob_unsafe)
            probs_safe = np.array(probs_safe)
            probs_unsafe = np.array(probs_unsafe)
            np.save('cache/llamaguard_probs_safe.npy', probs_safe)
            np.save('cache/llamaguard_probs_unsafe.npy', probs_unsafe)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aug', action='store_true', help='Whether to use prompt augmentation')
    args = parser.parse_args()
    inference(args)


