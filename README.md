# Guyu (谷雨)
pre-training and fine-tuning framework for text generation

#### Pre-training:

```
./prepare_data.sh
```

```
./train.sh
```

```
./inference.sh
```

#### Fine-tuning
Example: chat-bot

```
cd chat_bot
./fine_tune.sh
./inference.sh
```

#### Web Api:
```
./deploy.sh
```

#### Pre-trained models
- 12-layer, 768-hidden, 12-heads, Chinese (News + zhwiki, 200G) and English (Gigawords + Bookscorpus + enwiki, 60G) 

- 24-layer, 768-hidden, 12-heads, Chinese (News + zhwiki, 200G) and English (Gigawords + Bookscorpus + enwiki, 60G) 
