import os, sys
from pathlib import Path
from multiprocessing import Pool
from collections import Counter
import sys, re
import argparse

BUFSIZE = 100000
MAX_LEN = 256
MIN_LEN = 10

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--tgt_file', type=str)
    parser.add_argument('--bpe_model', type=str)
    parser.add_argument('--nprocessors', type=int)
    return parser.parse_args()

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def process(doc):
    # doc: one document with list of sentes
    #return sequence < max_len
    xs = []
    if not doc:
        return xs
    for sent in doc:
        ws = sent.split()
        if len(ws) > MAX_LEN:
            segs = chunks(ws, MAX_LEN)
            for seg in segs:
                xs.append(seg)
        else:
            xs.append(ws)
    res = []
    xi = []
    for i in range(len(xs)):
        ws = xs[i]
        if len(xi) + len(ws) <= MAX_LEN:
            xi.extend(ws)
        else:
            res.append(xi)
            xi = []
            #i -= 2
            xi.extend(ws)
            
    if xi and len(xi) >= MIN_LEN:
        res.append(xi)
    return res

def save(cnt, docs, nprocessors, fo):
    res = pool.map(process, docs, len(docs)//nprocessors)
    all_lines = []
    for xs in res:
         all_lines.extend(xs)
    
    for x in all_lines:
        cnt.update(x)
        fo.write(' '.join(x)+'\n')

if __name__ == "__main__":
    print("start..")
    args = parse_config()

    pool = Pool(args.nprocessors)
    cnt = Counter()
    docs = []
    path = Path(args.tgt_file)
    parent_path = path.parent
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    with open(args.tgt_file, 'w', encoding ='utf8') as fo:
        with open(args.src_file, "r") as fi:
            doc = []
            for line in fi:
                line = line.strip()
                if line:
                    doc.append(line)
                else:
                    docs.append(doc)
                    doc = []
                    
                if len(docs) == BUFSIZE:
                    save(cnt, docs, args.nprocessors, fo)                    
                    docs = []
                    print(BUFSIZE)
        if doc:
            docs.append(doc)
            save(cnt, docs, args.nprocessors, fo)
            print(len(docs))

    print("vocab")
    with open(str(parent_path)+ '/vocab.txt', 'w', encoding ='utf8') as f:
        for x, y in cnt.most_common():
            f.write(x + '\t' + str(y) + '\n')
    print("done")
