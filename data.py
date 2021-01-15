import random
import torch
import numpy as np

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
LS, RS, SP = '<s>', '</s>', ' '
BUFSIZE = 4096000

def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len - len(x))
        else:
            y = x + [0]*(max_len - len(x))
        ys.append(y)
    return ys

def batchify(data, vocab):
    truth, inp, msk = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        msk.append([1 for i in range(len(x) -1)])

    truth = torch.LongTensor(ListsToTensor(truth, vocab)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk

def s2t(strs, vocab):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk

def s2xy(lines, vocab, max_len, min_len):
    data = parse_lines(lines, max_len, min_len)
    return batchify(data, vocab)  

def parse_lines(lines, max_len, min_len):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(tokens) >= min_len:
            data.append(tokens)
    return data

class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = parse_lines(lines[:-1], self.max_len, self.min_len) #the last sent may be imcomplete
        random.shuffle(data)
        
        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK, BOS, EOS] + [LS, RS, SP] + (specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)
