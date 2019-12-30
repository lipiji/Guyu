import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from biglm import BIGLM
from data import Vocab, DataLoader, s2t

gpu = 0
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

print("loading...")
m_path = "./model/12L_10G.ckp"
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./model/12L_10G.vocab.txt")
print("done.")

MAX_LEN = 200

k = 40
def top_k_inc(s):
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(MAX_LEN):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
            else:
                logits = probs[0, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        s = [sent + t for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for i in s:
        print(i)

def top_k(s):
    start = time.time()
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(MAX_LEN):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            logits = probs[len(s[i]) - 1, i]
            ps, idx = torch.topk(logits, k=k)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        s = [sent + t for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for i in s:
        print(i)
    print(time.time()-start)


p = 0.95
def top_p_sampling(logits):
    ps, idx = torch.topk(logits, k=k)
    for i in range(2*k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_p(s):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(MAX_LEN):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            logits = probs[len(s[i]) - 1, i]
            ps, idx = top_p_sampling(logits)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        s = [sent + t for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for i in s:
        print(i)

g = 10
def top_g_sampling(logits):
    ps, idx = torch.topk(logits, k=k)
    for i in range(g, k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_g(s):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(MAX_LEN):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            logits = probs[len(s[i]) - 1, i]
            ps, idx = top_g_sampling(logits)
            ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        s = [sent + t for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for i in s:
        print(i)

def greedy(s):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(MAX_LEN):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            next_tk.append(lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))
        s = [sent + t for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for i in s:
        print(i)

def beam_decode(s, x, lm_vocab):
    beam_size = 5
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).cuda(gpu)

    x = x.cuda(gpu)
    ys = x.unsqueeze(1)

    for step in range(MAX_LEN):
        y_pred, _ = lm_model.work(ys)
        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_y_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size)[1]

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size))
        ys_now = []
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            ys_now.append(copy.copy(ys[:,j]))
        ys = torch.stack(ys_now, dim = 1) 

        last_traces = []
        last_scores = []
        for i in range(len(traces_now)):
            last_traces.append(traces_now[i])
            last_scores.append(scores_now[i])
        
        last_scores = torch.FloatTensor(np.array(last_scores).reshape((beam_size, 1))).cuda(gpu)
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            next_y.append(eid)
        next_y = np.array(next_y).reshape((1, beam_size))
        next_y = torch.LongTensor(next_y).cuda(gpu)
        
        ys = torch.cat([ys, next_y], dim=0)
        
        # end for loop

    for i in range(beam_size):
        samples.append([str(e.item()) for e in last_traces[i]])
        sample_scores[i] = last_scores[i]
    
    #weight by length
    for i in range(len(sample_scores)):
        sent_len = float(len(samples[i]))
        lp = np.power(5 + sent_len, 0.9) / np.power(5 + 1, 0.9)
        #sample_scores[i] /= lp
    
    idx_sorted_scores = np.argsort(sample_scores) # ascending order

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) > 0:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    dec_words = []
    for sample in sorted_samples[::-1]:
        for e in sample:
            e = int(e)
            dec_words.append(lm_vocab.idx2token(e))
        print(s + ''.join(dec_words))
        dec_words = []
        break

    #print()

def beam_search(s, lm_vocab):
    x, m = s2t(s, lm_vocab)
    for i in range(len(s)):
        beam_decode(s[i], x[:len(s[i]), i], lm_vocab)
        #break

s = ["丕子"]
print("\ntop_k (k="+str(k)+"):")
top_k_inc(s)
    
