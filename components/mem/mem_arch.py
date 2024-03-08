from sklearn.feature_extraction.text import CountVectorizer
import pinecone, os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import PineconeApiException

weights = {}
load_dotenv()
class mem_arch:
    def __init__(self, input_text):
        self.input_text = input_text
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit([input_text])
        self.vocab = self.vectorizer.get_feature_names_out()
        self.word_counts = self.vectorizer.transform([input_text]).toarray().flatten()
        self.mem = []

    def process_word_vectors(self):
        for word, count in zip(self.vocab, self.word_counts):
            vec = [0] * len(self.vocab)
            vec[self.vectorizer.vocabulary_[word]] = count
            self.mem.append([word] + vec)

    def cron_weight(self): 
        n = len(self.mem)
        """ 
        Attributes chonrological weights to word vectors where:
        Priotiry 1 has weight n 
        Priotirty 2 has weight n - 1
        ...
        Priotirty n has weight 1
        """
        for i, item in enumerate(self.mem):
            if i == 0:
                weights[tuple(item)] = n
            elif i == len(self.mem) - 1:
                weights[tuple(item)] = 1
        return weights

    def cache(self, index_name='eisa-epi-mem', namespace='ns1'):
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=PINECONE_API_KEY)
        try:
            indexes = pc.list_indexes()
            if index_name not in indexes:
                print(f"Index '{index_name}' does not exist, creating a new index...")
                pc.create_index(name=index_name, dimension=1536, metric="cosine",
                                spec=ServerlessSpec(cloud="aws", region="us-west-2"))
            else:
                print(f"Using existing index '{index_name}'.")
        except PineconeApiException as e:
            if e.status == 409 and "ALREADY_EXISTS" in str(e):
                print(f"Index '{index_name}' already exists.")
            else:
                print(f"Error handling index creation: {e}")
                return
        index = pc.Index(index_name)
        for item in self.mem:
            word, vector = item[0], item[1:]
            vector_dict = {
            "id": word,
            "values": vector
            }
            print(f"Vector Dictionary: {vector_dict}")
            try:
                index.upsert([vector_dict], namespace=namespace)
            except PineconeApiException as e:
                print(f"Error upserting vector for word '{word}': {e}")
                continue
        print("Word vectors cached successfully!")


input_text = "what is a human computer interface"
word_vector_processor = mem_arch(input_text)
word_vector_processor.process_word_vectors()
print(word_vector_processor.mem)
print(word_vector_processor.cron_weight())
word_vector_processor.cache()
