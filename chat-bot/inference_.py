import sys
sys.path.append("../")
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time
from inference import *

mstime = lambda: int(round(time.time() * 1000))    

device = 0
print("loading...")
device = 0
m_path = "./ckpt/epoch0_batch_3999"
v_path = "../model/12L_10G.vocab.txt"
lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
print("done.")

max_len = 50
qs = ["你看庆余年了么？", "我爱你！"]
i = 0
for q in qs:
    start = mstime()
    i += 1
    s = [[w for w in q]+ ["<bos>"]]

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
    print("bm5: ", r2)
    print("tk5: ", r3)
    print("tk10: ", r4)
    print("tk20: ", r5)
    print("tk50: ", r6)
    print("tk500: ", r7)
    print("tp0.9: ", r8)
    print(mstime()-start)
