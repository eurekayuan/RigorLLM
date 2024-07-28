import torch
import torch.nn.functional as F
import json
import os
import nltk
from nltk import tokenize
import torch
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher



def embed_inputs(embedding, logits, x_onehot=None, z_onehot=None, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1).type(torch.float16)
    # embedding : [vocab_size, embedding_size]
    # logits:     [batch_size, length, vocab_size]
    if x_onehot is not None:
        probs = torch.cat((x_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)), dim=1)
    if z_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), z_onehot.type(torch.FloatTensor)), dim=1)
        
    probs = probs.to(device)
    return torch.matmul(probs, embedding)


def embed_inputs_target(embedding, logits, x_onehot=None, z_onehot=None, target_onehot=None, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1).type(torch.float16)
    # embedding : [vocab_size, embedding_size]
    # logits:     [batch_size, length, vocab_size]
    if x_onehot is not None:
        probs = torch.cat((x_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)), dim=1)
    if z_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), z_onehot.type(torch.FloatTensor)), dim=1)
    if target_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), target_onehot.type(torch.FloatTensor)), dim=1)
    probs = probs.to(device)
    return torch.matmul(probs, embedding)


def _greedy(logits):
    _, last = torch.topk(logits, k=1, dim=-1)
    return last


def top_k_filter_3d(logits, k, probs=False, mask=None, extra_mask=None):
    """
    logits.shape = [batch_size, length, vocab_size]
    extra_mask: [batch_size, length, vocab_size], 1 if reserve
    """
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        if mask is None:
            _, indices = torch.topk(logits, k)
            mask = torch.zeros_like(logits).scatter_(2, indices, 1)
        if extra_mask is not None:
            mask = ((mask + extra_mask) > 0).float()
        if probs:
            return logits * mask
        return logits * mask + -BIG_CONST * (1-mask)


def top_k_filter(logits, k, probs=False):
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def _topk(logits, k=10):
    logits = top_k_filter(logits, k)
    probs = F.softmax(logits, dim=-1)
    last = torch.multinomial(probs, num_samples=1)
    return last


def top_p(logits, thres = 0.5, filter_value=-float('Inf')):
    assert len(logits.shape) == 1
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > thres
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits


def get_text_from_logits(logits, tokenizer):
    output_so_far = None
    last = None
    logp = 0
    for i in range(logits.shape[1]):
        last = _greedy(logits[:, i, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1).data.cpu().numpy()[:, last.data.cpu().numpy()]

    nll = -logp
    batch_size = output_so_far.shape[0]
    text = []
    for i in range(batch_size):
        text_i = tokenizer.decode(output_so_far[i].tolist())
        text_i = text_i.replace('\n', ' ')
        text_i += ". "
        text.append(text_i)

    return text, nll, output_so_far


def get_text_from_logits_topk(logits, tokenizer, top_k=1):
    output_so_far = None
    last = None
    logp = 0

    for i in range(logits.shape[1]):
        last = _topk(logits[:, i, :], top_k)
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1)[:, last.item()].item()

    nll = -logp
    text = tokenizer.decode(output_so_far.tolist()[0])
    text = text.replace('\n', ' ')
    return text, nll, output_so_far


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    onehot.to(tensor.device)
    return onehot


def initialize(model, x, length, temperature, batch_size, device, tokenizer):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    output = model.generate(x, max_length=length + x.shape[-1], do_sample=True, top_k=10)
    logits = model(output).logits
    logits_so_far = logits[:, -(length+1):-1, :] / temperature
        
    return logits_so_far


def decode_with_model_topk(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None):
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    embeddings_weight =  model.get_input_embeddings().weight
    input_embeds = torch.matmul(x_onehot.float().to(embeddings_weight.dtype), embeddings_weight)
    mask_t_all = None
    logits_so_far = None

    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds, use_cache=True)
        past = model_outputs.past_key_values    
        logits_t = model_outputs.logits[:, -1:, :]
        assert logits_t.shape[1] == 1, logits_t.shape   
        _, indices_t = torch.topk(logits_t, topk)
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1).to(embeddings_weight.dtype), embeddings_weight)
    return get_text_from_logits(
        top_k_filter_3d(y_logits, topk, mask=mask_t_all, extra_mask=extra_mask),
        tokenizer)


def soft_forward_loss(model, y_logits, topk, x_onehot, x_past, extra_mask=None):
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        device=x_onehot.device
    )
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds).logits
    x_length = x_onehot.shape[1]
    y_logits = xy_logits[:, x_length - 1:-1, :]

    _, indices_t = torch.topk(y_logits, topk)
    mask_t_all = torch.zeros_like(y_logits).scatter_(2, indices_t, 1)

    logp = F.log_softmax(y_logits, dim=-1)
    p = mask_t_all

    return -(p * logp).sum(dim=-1).mean(dim=-1)


def soft_backward_loss(model, y_logits_, yz_logits_rev, topk):
    embeddings_weight = model.get_input_embeddings().weight[1:yz_logits_rev.shape[-1]+1]
    y_embeds = embed_inputs(
        embeddings_weight,
        yz_logits_rev,
        device=yz_logits_rev.device
    )
    y_logits = model(inputs_embeds=y_embeds).logits
    yz_logits_rev_rev_t = torch.flip(y_logits, [1])
    yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]   
    yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]
    
    tmp_logits = yz_logits_rev_rev_t_
    repetition_mask = torch.cat([F.softmax(tmp_logits[:, 1:, :], dim=-1),
                                 torch.zeros_like(tmp_logits[:, -1:, :])], dim=1)
    yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4

    _, indices_t = torch.topk(yz_logits_rev_rev_t_, topk)
    mask_t_all = torch.zeros_like(yz_logits_rev_rev_t_).scatter_(2, indices_t, 1)
    logp = F.log_softmax(yz_logits_rev_rev_t_, dim=-1)
    p = mask_t_all
    
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def post_process(text_ids, model, max_length, length, tokenizer, device):
    text_ids_complete = sentence_completion(text_ids, model, max_length, device)
    batch_size = text_ids.shape[0]
    text_so_far_all = []
    for bi in range(batch_size):
        text_complete = tokenizer.decode(text_ids_complete[bi].tolist())
        text_complete = text_complete.replace('\n', ' ')
        sents = nltk.sent_tokenize(text_complete)
        text_so_far = None
        length_so_far = 0
        for i, sent in enumerate(sents):
            text_so_far = sent if text_so_far is None else text_so_far + ' ' + sent
            sent_length = len(sent.split())
            length_so_far += sent_length
            if length_so_far >= length:
                break
        text_so_far += ". "
        text_so_far_all.append(text_so_far)
    return text_so_far_all


def sentence_completion(text_ids, model, max_length, device):
    output_so_far = text_ids
    past = None
    last_embeds = None
    for i in range(max_length - text_ids.shape[1]):
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            last_embeds = model.get_input_embeddings()(last)

            if output_so_far.shape[1] > 1:
                model_outputs = model(output_so_far[:, :-1])
                past = model_outputs.past_key_values

        model_outputs = model(past_key_values=past, inputs_embeds=last_embeds, use_cache=True)
        logits = model_outputs.logits
        past = model_outputs.past_key_values

        last = _greedy(logits[:, -1, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        last_embeds = model.get_input_embeddings()(last)
    return output_so_far


def soft_distance(logits_perturbed, logits):
    return torch.nn.MSELoss()(logits_perturbed, logits)


def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def soft_nll_detach(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1).detach()
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean()


def additional_nll(logits, cur_text_ids):
    return torch.nn.CrossEntropyLoss()(
        logits.view(-1, logits.shape[-1]),
        cur_text_ids.view(-1)
    )


def soft_forward(model, x_onehot, y_logits, extra_mask=None, x_past=None, detach=True):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        device=y_logits.device
    )
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds, use_cache=True).logits
    if x_onehot != None:
        x_length = x_onehot.shape[1]
        y_logits = xy_logits[:, x_length - 1:-1, :]
    else:
        x_length = 1
        y_logits = xy_logits
    if detach:
        return y_logits.detach()
    else:
        return y_logits


def soft_forward_xyz(model, x_onehot, y_logits, z_onehot):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xyz_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        z_onehot=z_onehot,
        device=y_logits.device
    )
    xyz_logits = model(inputs_embeds=xyz_embeds).logits
    if x_onehot is not None:
        xy_length = x_onehot.shape[1] + y_logits.shape[1]
    else:
        xy_length = y_logits.shape[1]
    return xyz_logits, xy_length


def soft_backward(model, y_logits_rev):
    embeddings_weight = model.get_input_embeddings().weight[1:y_logits_rev.shape[-1]+1]
    y_embeds = embed_inputs(
        embeddings_weight,
        y_logits_rev,
        device=y_logits_rev.device
    )
    y_logits_ = model(inputs_embeds=y_embeds).logits
    return y_logits_[:, :-1, :]


def pre_filter(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None):
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    mask_t_all = None
    logits_so_far = None
    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds)
        past = model_outputs.past_key_values
        logits_t = model_outputs.logits[:, -1:, :]
        assert logits_t.shape[1] == 1, logits_t.shape
        _, indices_t = torch.topk(logits_t, topk)
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)
    return get_text_from_logits(
        top_k_filter_3d(y_logits, topk, mask=mask_t_all, extra_mask=extra_mask),
        tokenizer)


def collect_json_lines(model_output_json_file):
    with open(model_output_json_file, 'r') as fr:
        lines = fr.readlines()
        json_lines = [json.loads(x.strip()) for x in lines]
        return json_lines


def post_sent(text_complete):
    sents = nltk.sent_tokenize(text_complete)
    sent = ' '.join(sents[0].strip().split())
    return sent


def _has_repeat_sent(hyp):
    """
    Detect if the sentences in `hyp` are repeat.
    Args:
        hyp: A list of three sentences.
    """
    if len(hyp) <= 1:
        return False

    for i in range(1, len(hyp)):
        a = hyp[i-1]
        b = hyp[i]

        if a == b:
            return True

        s = SequenceMatcher(None, a, b)
        if len(a) > 5 and len(b) > 5 and s.ratio() >= 0.85:
            return True

    return False


def _has_repeat_substring(s, MINLEN=4, MINCNT=4):
    d = {}
    has_repeat = False
    for sublen in range(int(len(s)/MINCNT)-1, MINLEN-1, -1):
        for i in range(0, len(s)-sublen):
            sub = s[i:i+sublen]
            if len(sub.strip()) < sublen:
                continue
            cnt = s.count(sub)
            if cnt >= MINCNT and sub not in d:
                 d[sub] = cnt
                 # print('repeat_substring: |' + sub + '| in |' + s + '|')
                 has_repeat = True
                 break
        if has_repeat:
            break
    return has_repeat


def has_repeat(sents_for_substr):
    """
    Detect if the hypothesis text has repeat patterns.
    """
    has_repeat_substring = False
    for h in sents_for_substr:
        has_repeat_substring = has_repeat_substring or _has_repeat_substring(h) or _has_repeat_substring(h, MINLEN=20, MINCNT=2)
    return has_repeat_substring


def compute_loss(model, tokenizer, device, x="", z="", y="", constraints=None, args=None, model_back=None, zz=None):
    '''
    x: left context   (prompt in lexical constrained task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    batch_size = 2

    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(batch_size, 1)
    x_onehot = x_onehot.repeat(batch_size, 1, 1)

    z_ = tokenizer.encode(z)[1:] # delete the "." token we appended before
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_t = z_t.unsqueeze(0).repeat(batch_size, 1)

    y_ = tokenizer.encode(y)[1:] # delete the "." token we appended before
    y_t = torch.tensor(y_, device=device, dtype=torch.long)
    y_onehot = one_hot(y_t, dimension=tokenizer.vocab_size)
    y_onehot = y_onehot.repeat(batch_size, 1, 1)
    y_t = y_t.unsqueeze(0).repeat(batch_size, 1)

    y_logits_ = y_onehot / 0.0001

    c_loss = batch_log_bleulosscnn_ae(
        decoder_outputs=y_logits_.transpose(0, 1),
        target_idx=z_t,
        ngram_list=[2, 3]
    )

    return c_loss.mean().item()


def _get_adverbs_and_nnps(z_words):
    pos = nltk.pos_tag(z_words)
    adverbs = [w[0] for w in pos if 'RB' in w[1]]
    nnps = [w[0] for w in pos if 'NNP' in w[1]]
    return adverbs, nnps


def _get_keywords(z, x, args):
    stop_words = set(stopwords.words('english'))
    z_words = word_tokenize(z)
    z_adverbs, z_nnps = _get_adverbs_and_nnps(z_words)
    ret_words = []
    for w in z_words:
        if w in z_nnps:
            if w not in ret_words:
                ret_words.append(w)
        else:
            w = w.lower()
            if w not in stop_words and w.isalnum() and w not in z_adverbs and w not in ret_words:
                ret_words.append(w)

    if args.abductive_filterx:
        x_words = word_tokenize(x)
        ret_words = [w for w in ret_words if w not in x_words]

    return ' '.join(ret_words)


def get_similarity(model, y_logits, ref_vec):
    y_embs = embed_inputs(model.get_input_embeddings().weight, y_logits / 0.001, device=y_logits.device)
    y_vec = model(inputs_embeds=y_embs, use_cache=True, output_hidden_states=True).hidden_states[-1].mean(dim=1)
    return F.cosine_similarity(ref_vec, y_vec)


def get_embedding_from_text(model, tokenizer, ref, device):
    ref_ = tokenizer.encode(ref)[:]
    ref_ = torch.tensor(ref_).to(device)
    if ref_.dim() == 1:
        ref_ = ref_.unsqueeze(0)
    ref_vec = model(ref_, output_hidden_states=True).hidden_states[-1].mean(dim=1).detach()
    return ref_vec
