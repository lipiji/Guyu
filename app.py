import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import logging


from biglm import BIGLM
from data import Vocab, DataLoader, s2t
    
from flask import Flask,request
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

gpu = 1
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    #lm_args.dropout = 0.1
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    return lm_model, lm_vocab, lm_args

m_path = "./ckpt/epoch0_batch_1509999" #1449999
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./data/vocab.txt")

lm_model.eval()

MAX_LEN = 80

k = 40

def top_k(s, max_len):
    incremental_state = None 
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    for l in range(max_len):
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
    return s


@app.route("/gpt")
def gen():
    q = request.args.get("q")
    if not q:
        q = " "
    logging.info("Query = " + q)
    max_len = int(request.args.get("len"))
    res = top_k([q], max_len)[0]
    end_sybs = ["?", "!", "。", "？", "！"]
    end_idx = 0
    for w in end_sybs:
        ridx = res.rfind(w)
        if ridx > end_idx:
            end_idx = ridx
    if end_idx > 0:
        res = res[:end_idx + 1]
    return res

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
