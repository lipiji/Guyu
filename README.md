# Guyu (谷雨)
pre-training and fine-tuning framework for text generation

backbone code for "An Empirical Investigation of Pre-Trained Transformer Language Models for Open-Domain Dialogue Generation": https://arxiv.org/abs/2003.04195
```
@article{DBLP:journals/corr/abs-2003-04195,
  author    = {Piji Li},
  title     = {An Empirical Investigation of Pre-Trained Transformer Language Models
               for Open-Domain Dialogue Generation},
  journal   = {CoRR},
  volume    = {abs/2003.04195},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.04195},
  archivePrefix = {arXiv},
  eprint    = {2003.04195},
  timestamp = {Tue, 10 Mar 2020 13:33:48 +0100}}
}
```

torch>=1.0

#### Pre-training:

```
./prepare_data.sh
./train.sh
./inference.sh
```

#### Fine-tuning
Example: chat-bot

```
cd chat_bot
./prepare_data.sh
./fine_tune.sh
./inference.sh
```

#### Web Api:
```
./deploy.sh
```

#### Pre-trained models
- 12-layer, 768-hidden, 12-heads, Chinese (News + zhwiki, 200G) and English (Gigawords + Bookscorpus + enwiki, 60G) 

- 24-layer, 1024-hidden, 16-heads, Chinese (News + zhwiki, 200G) and English (Gigawords + Bookscorpus + enwiki, 60G) 

- download them: https://github.com/lipiji/Guyu/tree/master/model

#### References:
- GPT2: https://openai.com/blog/better-language-models/
- https://github.com/jcyk/BERT
