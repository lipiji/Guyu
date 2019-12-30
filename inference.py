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

def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

def top_k_inc(lm_model, lm_vocab, device, s, k, max_len):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
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
        x = x.to(device)
        bidx = torch.ByteTensor(bidx).to(device)
        incremental_state["bidx"] = bidx
    res += s_
        
    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>")[1]
    else:
        return r

def top_p_sampling(logits, k, p):
    ps, idx = torch.topk(logits, k=k)
    for i in range(k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_p_inc(lm_model, lm_vocab, device, s, k, p, max_len):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = top_p_sampling(logits, k, p)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = top_p_sampling(logits, k, p)
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
        x = x.to(device)
        bidx = torch.ByteTensor(bidx).to(device)
        incremental_state["bidx"] = bidx
    res += s_
    
    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>")[1]
    else:
        return r

g = 10
def top_g_sampling(logits):
    ps, idx = torch.topk(logits, k=k)
    for i in range(g, k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx

def top_g(lm_model, lm_vocab, device, s, max_len):
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    for l in range(max_len):
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
        x = x.to(device)

    for i in s:
        print(i)

def greedy(lm_model, lm_vocab, device, s, max_len):
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
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
        x = x.to(device)
    res += s_
        
    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>")[1]
    else:
        return r


def beam_decode(lm_model, lm_vocab, device, s, x, max_len):
    beam_size = 5
    
    num_live = 1
    num_dead = 0 
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(device)

    x = x.to(device)
    ys = x.unsqueeze(1)

    for step in range(max_len):
        y_pred, _ = lm_model.work(ys)

        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_y_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]

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

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(device)
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            next_y.append(eid)
        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(device)
        
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

def beam_search(lm_model, lm_vocab, device, s, max_len):
    x, m = s2t(s, lm_vocab)
    return beam_decode(lm_model, lm_vocab, device, s[0], x[:len(s[0]), 0], max_len)

if __name__ == "__main__":
    device = 0
    print("loading...")
    m_path = "./model/12L_10G.ckpt"
    v_path = "./model/12L_10G.vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")
    
    max_len  = 50
    qs = ["庆余年", "我爱你"]
    print(qs)
    i = 0
    for q in qs:
        start = mstime()
        i += 1
        s = [[w for w in q]]

        r1 = greedy(lm_model, lm_vocab, device, s, max_len)

        r2 = beam_search(lm_model, lm_vocab, device, s, max_len)

        r3 = top_k_inc(lm_model, lm_vocab, device, s, 5, max_len)

        r4 = top_k_inc(lm_model, lm_vocab, device, s, 10, max_len)

        r5 = top_k_inc(lm_model, lm_vocab, device, s, 20, max_len)

        r6 = top_k_inc(lm_model, lm_vocab, device, s, 50, max_len)

        r7 = top_k_inc(lm_model, lm_vocab, device, s, 500, max_len)

        r8 = top_p_inc(lm_model, lm_vocab, device, s, 20, 0.95, max_len)
    
        print(i)
        print("q: ", q)
        print("greedy: ", r1)
        print("bm5: ", q+r2)
        print("tk5: ", r3)
        print("tk10: ", r4)
        print("tk20: ", r5)
        print("tk50: ", r6)
        print("tk500: ", r7)
        print("tp0.95: ", r8)
        print(mstime()-start)
    
