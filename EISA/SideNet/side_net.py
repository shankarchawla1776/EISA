import torch 
import torch.nn as nn
import torch.nn.functional as F
from gensim.test.utils import common_texts, common_dictionary, common_corpus
from gensim.models import Word2Vec
import pandas as pd 

# create unique corpus 

input = "science"
spl = input.split()

model = Word2Vec(common_texts, min_count=1)
vocab = set(model.wv.index_to_key)

filt = [word for word in spl if word in vocab]

data = []

for word in filt:
    vec = model.wv[word]
    data.append([word] + vec.tolist())

print(data)

columns = ["word"] + [f"vec_{i}" for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)
df.to_csv("word_vectors.csv", index=False)


