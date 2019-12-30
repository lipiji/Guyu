import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from biglm import BIGLM
from data import Vocab, DataLoader, s2t



def init_seeds():
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

#init_seeds()

gpu = 1
def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    return lm_model, lm_vocab, lm_args

m_path = "./ckpt/epoch0_batch_1449999"
lm_model, lm_vocab, lm_args = init_model(m_path, gpu, "./data/train.txt_vocab")

lm_model.eval()

MAX_LEN = 150

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
       
        '''
        if step == 0:
            o_len = ys.size(0)
            for i in range(o_len-1):
                last_scores += torch.log(y_pred[i, 0, ys[i, 0]])
        '''

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

#s = ["推水晶,兵线", "马可装备末世", "诸葛亮对线嬴政", "干将莫邪", "复活甲", "貂蝉"]
#s = ["史树明", "刘晓江", "李丕绩", "闭玮", "王琰", "李华阳", "王龙跃", "王星", "涂兆鹏", "刘乐茂", "黄国平", "韩家龙", "李菁", "张海松"]
#s = ["机器学习", "腾讯", "中国独角兽会讲汉语"]
#s = ["gpt2越来越牛逼了", "锅在天上飞", "你才神经病", "这道题怎么解", "关于姐弟恋,大家有神马想说的", "停车坐爱枫林晚", "我觉得男人做饭的时候特别性感", "屌丝终有逆袭日"]
#s = ["推水晶", "干将莫邪", "诸葛亮", "花木兰", "李元芳", "貂蝉", "马可", "打野", "张飞大招"]
#s = ["林伟", "李丕绩", "石贝", "徐引擎", "邴立东", "三哥", "李昕", "香港中文大学"]
#s = ["腾讯副总裁姚星", "姚总", "张老师", "俞老师", "史老师", "刘晓江"]
#s = ["腾讯AI Lab", "诸葛亮一技能", "东皇大招", "盾山二技能"]
#s = ["山有木兮木有枝，", "人生若只如初见，", "苟利国家生死以，", "明月几时有？", "明月别枝惊鹊，"]
s = ["腾讯CEO马化腾"]

#print("\ngreedy:")
#greedy(s)

#print("\nbeam search:")
#beam_search(s, lm_vocab)

print("\ntop_k (k="+str(k)+"):")
#top_k(s)
top_k_inc(s)
#print("\ntop_p (p="+str(p)+")")
#top_p(s)
    
#print("\ntop_pg (["+str(g)+", " + str(k)+ "])")
#top_g(s)
