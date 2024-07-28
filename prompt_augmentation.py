import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from tqdm import tqdm
import numpy as np
import random
from datasets import load_dataset
import openai
client = openai.OpenAI()



def chatgpt(prompt, gpt4=False, temperature=0.0):
    model = "gpt-4" if gpt4 else "gpt-3.5-turbo"
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response


def preprocess_openai(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()  
    malicious_num = 0
    for line in lines:
        data = json.loads(line)
        instance = data['prompt']
        instances.append(instance)
        category = 0
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                category = 1
                malicious_num += 1
                break
        categories.append(category)
    print('Preprocessed openai. Total: {}. Malicious: {}.'.format(len(instances), malicious_num))
    return instances, categories


def toxic_chat_test():
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    random.seed(0)
    np.random.seed(0)
    indicies = range(len(dataset["test"]))
    # subsampled_data = random.sample(indicies, 1000)
    subsampled_data = range(1000)
    test_data = [dataset['test']['user_input'][i] for i in subsampled_data]
    toxicity = [dataset['test']['toxicity'][i] for i in subsampled_data]
    print('Loaded {} test data, malicious: {}'.format(len(test_data), sum(toxicity)))
    return test_data, toxicity


outputs = []
# instances, categories = preprocess_openai()
instances, categories = toxic_chat_test()
for instance, category in tqdm(zip(instances, categories)):
    output = {
        'original': instance,
        'paraphrase': [],
        'summary': [],
        'malicious': category
    }
    prompt1 = "Paraphrase: \n{}".format(instance)
    for i in range(3):
        try:
            output1 = chatgpt(prompt1, temperature=0.8)
        except:
            output1 = 'None'
        prompt2 = "Summarize the following text into 1-2 sentences: \n{}".format(output1)
        try:
            output2 = chatgpt(prompt2, temperature=0.0)
        except:
            output2 = 'None'
        output['paraphrase'].append(output1)
        output['summary'].append(output2)
    outputs.append(output)
    with open('toxicchat_processed.json', 'w') as f: 
        json.dump(outputs, f)
    

