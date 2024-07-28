import os
import json
import random
from datasets import load_dataset
import numpy as np



def preprocess_hexphi_gen(root='data/hexphi'):
    subsets = os.listdir(root)
    cate2iotexts = {}
    for subset in subsets:
        cate = subset.split('eval_')[-1].split('_gpt_4')[0]
        with open(os.path.join(root, subset)) as f:
            lines = f.readlines()
        itexts = []
        otexts = []
        for line in lines:
            data = json.loads(line)
            itexts.append(data['user'])
            otexts.append(data['model'])
        cate2iotexts[cate] = (itexts, otexts, lines)
    return cate2iotexts


def preprocess_dataset_as_hexphi(root='data/hexphi'):
    subsets = os.listdir(root)
    instances, categories = [], []
    category_id = 2
    for subset in subsets:
        category = subset.split('eval_')[-1].split('_gpt_4')[0].split('.jsonl')[0]
        print(category)
        with open(os.path.join(root, subset)) as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if 'user' in data.keys():
                instance = (data['user'], category_id)
                instances.append(instance)
        category_id += 1
        categories.append(category + '_user')
        for line in lines:
            data = json.loads(line)
            if 'model' in data.keys():
                instance = (data['model'], category_id)
                instances.append(instance)
        category_id += 1
        categories.append(category + '_model')
    print('Preprocessed hexphi. Categories: {}. Total: {}.'.format(len(categories), len(instances)))
    return instances, categories


def preprocess_hotpot(root='data/hotpot', split='train', total=1000):
    instances = []
    file = 'hotpot_dev_fullwiki_v1.json' if split == 'train' else 'hotpot_test_fullwiki_v1.json'
    with open(os.path.join(root, file)) as f:
        all_data = json.load(f)
    for data in all_data[:total]:
        instance = data['question']
        instances.append(instance)
    print('Preprocessed hotpot. Total: {}.'.format(len(instances)))
    return instances


def preprocess_multiturn(root='data/multiturn'):
    instances = []
    with open(os.path.join(root, 'question.jsonl')) as f:
        lines = f.readlines()  
    for line in lines:
        data = json.loads(line)
        instance = data['turns'][0]
        instances.append(instance)
    print('Preprocessed multiturn. Total: {}.'.format(len(instances)))
    return instances


def preprocess_openai(root='data/openai'):
    instances, categories = [], []
    with open(os.path.join(root, 'samples-1680.jsonl')) as f:
        lines = f.readlines()
    cate2inst = {'none': []}
    for line in lines:
        data = json.loads(line)
        instance = data['prompt']
        malicious_flag = False
        for key in data.keys():
            if key != 'prompt' and data[key] != 0:
                malicious_flag = True
                if key not in cate2inst.keys():
                    cate2inst[key] = [instance]
                else:
                    cate2inst[key].append(instance)
        if not malicious_flag:
            cate2inst['none'].append(instance)
    red_team_data_eval = {}

    random.seed(0)
    for key, value in cate2inst.items():
        red_team_data_eval[key] = random.sample(value, 15)
        # red_team_data_eval[key] = value[:20]
    
    # Generate instances for validation and test data
    all_instances_eval = []
    for key, value in red_team_data_eval.items():
        all_instances_eval += value
    all_instances_eval = list(set(all_instances_eval))

    malicious_num = 0
    val_indices, test_indices = [], []
    for i, line in enumerate(lines):
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
        if instance in all_instances_eval:
            val_indices.append(i)
        else:
            test_indices.append(i)
    
    print('Preprocessed openai. Total: {}. Malicious: {}. Validation: {}. Test: {}.'.format(len(instances), malicious_num, len(val_indices), len(test_indices)))
    return instances, categories, val_indices, test_indices


def preprocess_openai_plus(root='openai_processed.json'):
    instances, categories = [], []
    with open(root) as f:
        data = json.load(f)
    malicious_num = 0
    for d in data:
        x_ori = d['original']
        category = d['malicious']
        instances.append(x_ori)
        categories.append(category * 3)
        for x_par in d['paraphrase']:
            instances.append(x_par)
            categories.append(category * 3 + 1)
        for x_sum in d['summary']:
            instances.append(x_sum)
            categories.append(category * 3 + 2)
        category = d['malicious']
        malicious_num += category * 7
        
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


def toxic_chat_train():
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    random.seed(0)
    np.random.seed(0)
    indicies = range(len(dataset["test"]))
    # subsampled_data = random.sample(indicies, 1000)
    subsampled_data = range(1000)
    test_data = [dataset['train']['user_input'][i] for i in subsampled_data]
    toxicity = [dataset['train']['toxicity'][i] for i in subsampled_data]
    print('Loaded {} train data, malicious: {}'.format(len(test_data), sum(toxicity)))
    return test_data, toxicity