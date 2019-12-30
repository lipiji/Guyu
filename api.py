import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import logging
from inference import *
    
from flask import Flask,request
app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

print("loading...")
device = 0
m_path = "./model/12L_10G.ckpt"
v_path = "./model/12L_10G.vocab.txt"
lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
print("done.")

@app.route("/guyu")
def gen():
    q = request.args.get("q")
    if not q:
        q = " "
    logging.info("Query = " + q)
    s = [[w for w in q]]
    res = top_p_inc(lm_model, lm_vocab, device, s, 50, 0.95, 100)
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
