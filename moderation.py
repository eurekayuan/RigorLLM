import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion, AgentType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
from argparse import ArgumentParser
from data_processing import *



def get_llm_embedding(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        last_hidden_state = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
    idx_of_the_last_non_padding_token = inputs.attention_mask.bool().sum(1)-1
    embedding = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token][0]
    return embedding


def get_embeddings(no_cache=False, embedder='cohere', training_data='hexphi', test_data='openai'):
    # Test data
    querys, test_categories, val_indices, test_indices = preprocess_openai()

    # Harmful data
    instances, category_names = preprocess_dataset_as_hexphi(root=training_data)
    training_data = training_data.split('/')[-1]  # for caching embeddings
    instances_text = [d[0] for d in instances]
    train_categories = [d[1] for d in instances]

    # Benign data
    instances_hotpot = preprocess_hotpot(split='train')
    instances_multiturn = preprocess_multiturn()
    instances_text += instances_hotpot
    instances_text += instances_multiturn
    train_categories += [0 for d in instances_hotpot]
    train_categories += [0 for d in instances_multiturn]
    
    category_names = ['benign', 'harmful_val'] + category_names

    if os.path.exists(f'cache/train_embedding_{embedder}_{training_data}.npy') and no_cache is False:
        train_embeddings = np.load(f'cache/train_embedding_{embedder}_{training_data}.npy')
        print('Loaded cached train embeddings')
    else:
        # Get LLM embeddings
        model_id = "meta-llama/LlamaGuard-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        train_embeddings = []
        for i, instance in enumerate(tqdm(instances_text)):
            if '_model' in category_names[train_categories[i]]:
                agent_type = AgentType.AGENT
            else:
                agent_type = AgentType.USER
            formatted_prompt = build_default_prompt(agent_type, create_conversation([instance]), LlamaGuardVersion.LLAMA_GUARD_1)
            embedding = get_llm_embedding(model, tokenizer, formatted_prompt)
            train_embeddings.append(embedding.cpu().numpy())
        train_embeddings = np.array(train_embeddings)
        np.save(f'cache/train_embedding_{embedder}_{training_data}.npy', train_embeddings)
        print('Generated train embeddings')
    
    if os.path.exists(f'cache/test_embedding_{embedder}_{test_data}.npy') and no_cache is False:
        test_embeddings = np.load(f'cache/test_embedding_{embedder}_{test_data}.npy')
        print('Loaded cached test embeddings')
    else:
        # Get LLM embeddings
        model_id = "meta-llama/LlamaGuard-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        test_embeddings = []
        for i, query in enumerate(tqdm(querys)):
            agent_type = AgentType.USER
            formatted_prompt = build_default_prompt(agent_type, create_conversation([query]), LlamaGuardVersion.LLAMA_GUARD_1)
            embedding = get_llm_embedding(model, tokenizer, formatted_prompt)
            test_embeddings.append(embedding.cpu().numpy())
        test_embeddings = np.array(test_embeddings)
        np.save(f'cache/test_embedding_{embedder}_{test_data}.npy', test_embeddings)
        print('Generated test embeddings')

    if os.path.exists(f'cache/test_embedding_{embedder}_{test_data}_plus.npy') and no_cache is False:
        test_embeddings_plus = np.load(f'cache/test_embedding_{embedder}_{test_data}_plus.npy')
        print('Loaded cached augmented test embeddings')
    else:
        querys_plus, test_categories_plus = preprocess_openai_plus()
        
        # Get LLM embeddings
        model_id = "meta-llama/LlamaGuard-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        test_embeddings_plus = []
        for i, query in enumerate(tqdm(querys_plus)):
            agent_type = AgentType.USER
            formatted_prompt = build_default_prompt(agent_type, create_conversation([query]), LlamaGuardVersion.LLAMA_GUARD_1)
            embedding = get_llm_embedding(model, tokenizer, formatted_prompt)
            test_embeddings_plus.append(embedding.cpu().numpy())
        test_embeddings_plus = np.array(test_embeddings_plus)
        np.save(f'cache/test_embedding_{embedder}_{test_data}_plus.npy', test_embeddings_plus)
        print('Generated augmented test embeddings')

    train_categories = np.array(train_categories)
    test_categories = np.array(test_categories)

    # Add validation data
    train_embeddings = np.concatenate([train_embeddings, test_embeddings[val_indices]], axis=0)
    train_categories = np.concatenate([train_categories, test_categories[val_indices]], axis=0)
    test_embeddings = test_embeddings[test_indices]
    test_categories = test_categories[test_indices]
    querys = [querys[i] for i in test_indices]

    assert train_embeddings.shape[0] == len(train_categories)
    assert test_embeddings.shape[0] == len(test_categories) == len(querys)
    print('Finished data processing. Training: {}. Test: {}.'.format(len(train_categories), len(test_categories)))

    return train_embeddings, train_categories, test_embeddings, test_categories, test_embeddings_plus, category_names, val_indices, test_indices


def get_scores(probs, gt_categories):
    pred = (np.array(probs) > 1 - np.array(probs)).astype(np.int32)
    accuracy, precision, recall, f1 = accuracy_score(gt_categories, pred), precision_score(gt_categories, pred), recall_score(gt_categories, pred), f1_score(gt_categories, pred)
    precisions, recalls, thresholds = precision_recall_curve(gt_categories, probs)
    area = auc(recalls, precisions)
    return area, accuracy, precision, recall, f1


def moderation_knn(args):
    harmful_categories = [
        # 'fraudulent_deceptive',
        # 'political',
        'physical_harm',
        # 'harm_children',
        # 'malware',
        # 'economic_harm',
        'hate_harass_violence',
        # 'financial_advice',
        # 'privacy_violation',
        'illegal_activity',
        'adult_content'
    ]

    train_embeddings, train_categories, test_embeddings, test_categories, test_embeddings_plus, category_names, val_indices, test_indices = get_embeddings(no_cache=args.no_cache, embedder=args.embedder, training_data=args.train_data, test_data=args.test_data)
    
    test_embeddings_plus = test_embeddings_plus.reshape(-1, 7, test_embeddings_plus.shape[-1])
    test_embeddings_plus = test_embeddings_plus[test_indices]

    harmful_indices = [1]
    for i, category_name in enumerate(category_names):
        for harmful_category in harmful_categories:
            if harmful_category in category_name:
                harmful_indices.append(i)
                break
    
    # Merge prediction with LLM
    probs_unsafe_llamaguard = np.load('cache/llamaguard_probs_unsafe.npy')[test_indices]
    probs_unsafe_llamaguard_plus = np.load('cache/llamaguard_probs_unsafe_plus.npy')
    probs_unsafe_llamaguard_plus = probs_unsafe_llamaguard_plus.reshape(-1, 7)
    probs_unsafe_llamaguard_plus = probs_unsafe_llamaguard_plus[test_indices]
    probs_unsafe_llamaguard_plus[:, 4:] = probs_unsafe_llamaguard_plus[:, 0][:, None]
    probs_unsafe_llamaguard_plus = np.mean(probs_unsafe_llamaguard_plus, axis=1)

    k = 16
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_embeddings, train_categories)
    
    probs = np.array(classifier.predict_proba(test_embeddings))
    probs_unsafe_knn = np.zeros_like(probs[:, 0])
    for i in harmful_indices:
        probs_unsafe_knn += probs[:, i]

    probs_unsafe_knn_plus = np.zeros_like(probs_unsafe_knn)
    for j in [0, 1, 2, 3, 4, 5, 6]:
        probs = np.array(classifier.predict_proba(test_embeddings_plus[:, j]))
        for i in harmful_indices:
            probs_unsafe_knn_plus += probs[:, i]
    probs_unsafe_knn_plus = probs_unsafe_knn_plus / 4

    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe_knn, test_categories)
    print('KNN                      AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))

    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe_knn_plus, test_categories)
    print('KNN + Prompt Aug         AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))

    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe_llamaguard, test_categories)
    print('Llama Guard              AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))
    
    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe_llamaguard_plus, test_categories)
    print('Llama Guard + Prompt Aug AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))

    alpha = 0.3
    probs_unsafe = alpha * probs_unsafe_knn + (1-alpha) * probs_unsafe_llamaguard
    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe, test_categories)
    print('Aggregation              AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))
    
    probs_unsafe = alpha * probs_unsafe_knn_plus + (1-alpha) * probs_unsafe_llamaguard_plus
    area, accuracy, precision, recall, f1 = get_scores(probs_unsafe, test_categories)
    print('Aggregation + Prompt Aug AUPRC: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(area, accuracy, precision, recall, f1))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--embedder', type=str, default='llamaguard')
    parser.add_argument('--train_data', type=str, help='train data should be organized in hexphi format')
    parser.add_argument('--test_data', type=str, default='openai')
    args = parser.parse_args()
    moderation_knn(args)