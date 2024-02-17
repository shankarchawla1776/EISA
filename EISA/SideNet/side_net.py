import torch 
import torch.nn as nn
import torch.nn.functional as F
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd 

input = "what is cognitive science?"
spl = input.split()
print(spl[0])
# print(common_texts)

model = Word2Vec(common_texts, min_count=1)
vocab = set(model.wv.index_to_key)

words_in_vocab = [word for word in spl if word in vocab]

# vec = model.wv['computer']
# print(vec)

for word in words_in_vocab:
    vec = model.wv.most_similar(word)
#     print(ms)

# print(ms)

# print(input)