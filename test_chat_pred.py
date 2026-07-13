import torch
import numpy as np
from chat import load_checkpoint
from dataload import load_data
from preprocess import tokenize, bag_of_words

samples = ["hii","hellow","Hello","Hi","hey there"]

model, vocab, tags = load_checkpoint('checkpoints/chatbot.pth')
intents = load_data()

for s in samples:
    tokens = tokenize(s)
    X = bag_of_words(tokens, vocab)
    X = torch.from_numpy(X).unsqueeze(0)
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        prob, pred = torch.max(probs, dim=1)
        print(s, '->', tags[pred.item()], f'(prob={prob.item():.3f})')
