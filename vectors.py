from pinecone import ServerlessSpec, Pinecone 
from dotenv import load_dotenv
from gensim.test.utils import common_texts
from word_vectors.word_vecs import model, vocab
import numpy as np
import os 

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(pinecone_api_key)



index = pc.Index("eisa")

def to_vec(sentence, model): 
    words = sentence.split()
    vecs = [] 
    for i in words: 
        try:
            vecs.append(model.wv[i])
        except KeyError: 
            pass 
    if vecs: 
        return np.mean(vecs, axis=0)
    else: 
        return None 
    
sentence = "what is a human computer interface"

vec = to_vec(sentence, model)
vec_form = np.round(vec, decimals=5)
index.upsert(item_ids=["prompt_1"], vectors=[vec])
print(vec_form)