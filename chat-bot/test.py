import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from biglm import BIGLM
from data import Vocab, DataLoader, s2t

mstime = lambda: int(round(time.time() * 1000))    

gpu = 0

def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    #lm_args.dropout = 0.1
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    return lm_model, lm_vocab, lm_args

m_path = "./ckpt/epoch12_batch_1289999"
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./model/vocab.txt")

lm_model.eval()

MAX_LEN = 50

k = 20

def top_k_inc(s, k):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
    res = []
    for l in range(MAX_LEN):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)
        bidx = torch.ByteTensor(bidx).to(gpu)
        incremental_state["bidx"] = bidx
    res += s_
        
    r = ''.join(res[0])
    return r.split("<bos>")[1]


def top_k(s):
    start = time.time()
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
    res = []
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
        
        s_ = []
        for sent, t in zip(s, next_tk):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)

    res += s_
        
    for i in res:
        print(''.join(i))

    print(time.time()-start)

p = 0.9
def top_p_sampling(logits):
    ps, idx = torch.topk(logits, k=k)
    for i in range(k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_p(s):
    res = []
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
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
            
        s_ = []
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)
    res += s_
        
    r = ''.join(res[0])
    return r.split("<bos>")[1]

def top_p_inc(s):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
    res = []
    for l in range(MAX_LEN):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = top_p_sampling(logits)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)
        bidx = torch.ByteTensor(bidx).to(gpu)
        incremental_state["bidx"] = bidx
    res += s_
        
    r = ''.join(res[0])
    return r.split("<bos>")[1]



g = 10
def top_g_sampling(logits):
    ps, idx = torch.topk(logits, k=k)
    for i in range(g, k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_g(s):
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
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
        s = [sent + [t] for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)

    for i in s:
        print(i)




def greedy(s):
    x, m = s2t(s, lm_vocab)
    x = x.to(gpu)
    res = []
    for l in range(MAX_LEN):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            next_tk.append(lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))
        
            
        s_ = []
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(gpu)
    res += s_
        
    r = ''.join(res[0])
    return r.split("<bos>")[1]


def beam_decode(s, x, lm_vocab):
    beam_size = 5
    
    num_live = 1
    num_dead = 0 
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(gpu)

    x = x.to(gpu)
    ys = x.unsqueeze(1)

    for step in range(MAX_LEN):
        y_pred, _ = lm_model.work(ys)

        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_y_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]
        
        '''
        ps, idx_top_joint_scores = torch.topk(cand_scores, 100)
        ps = F.softmax(ps)
        sampled = torch.multinomial(ps, num_samples = beam_size - num_dead)
        idx_top_joint_scores = idx_top_joint_scores[sampled]
        '''

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        ys_now = []
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            ys_now.append(copy.copy(ys[:,j]))


        num_live = 0  
        last_traces = []
        last_scores = []
        ys = []
        for i in range(len(traces_now)):
            w = lm_vocab.idx2token(traces_now[i][-1].item())
            if w == "<eos>":
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i] 
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                ys.append(ys_now[i])
                num_live += 1
        
        if num_live == 0 or num_dead >= beam_size:
            break
        ys = torch.stack(ys, dim = 1) 

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(gpu)
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            next_y.append(eid)
        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(gpu)
        
        ys = torch.cat([ys, next_y], dim=0)
       
        assert num_live + num_dead == beam_size 
        # end for loop

    if num_live > 0:
        for i in range(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1  

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

    res = []
    dec_words = []
    for sample in sorted_samples[::-1]:
        for e in sample:
            e = int(e)
            dec_words.append(lm_vocab.idx2token(e))
        r = ''.join(dec_words)
        #print(r)
        res.append(r)
        dec_words = []

    return res[0]


def beam_search(s, lm_vocab):
    x, m = s2t(s, lm_vocab)
    return beam_decode(s[0], x[:len(s[0]), 0], lm_vocab)


qs = ["你看庆余年了么？", "我爱你！"]

i = 0
for q in qs:
    start = mstime()
    i += 1
    s = [q.split()+ ["<bos>"]]

    r1 = greedy(s)

    r2 = beam_search(s, lm_vocab)

    r3 = top_k_inc(s, 5)

    r4 = top_k_inc(s, 10)

    r5 = top_k_inc(s, 20)

    r6 = top_k_inc(s, 50)

    r7 = top_k_inc(s, 500)

    r8 = top_p_inc(s)
    
    print(i)
    print("q: ", q)
    print("greedy: ", r1)
    print("bm5: ", r2)
    print("tk5: ", r3)
    print("tk10: ", r4)
    print("tk20: ", r5)
    print("tk50: ", r6)
    print("tk500: ", r7)
    print("tp0.9: ", r8)
    print(mstime()-start)
