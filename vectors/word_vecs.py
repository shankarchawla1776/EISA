from gensim.test.utils import common_texts, common_dictionary, common_corpus
from gensim.models import Word2Vec
import sys 


model = Word2Vec(common_texts, min_count=1)
vocab = set(model.wv.index_to_key)

print(sys.path)