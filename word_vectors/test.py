from sklearn.feature_extraction.text import CountVectorizer
import pinecone 
from dotenv import load_dotenv
import os 
from pinecone import Pinecone 

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key="PINECONE_API_KEY")
corpus = [
    "what is a brain computer interface",
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())

X_dense = X.toarray().tolist()

index = pinecone.Index(name="eisa", host="pinecone.io", api_key=PINECONE_API_KEY)

index.upsert(ids=['doc_1'], vectors=X_dense)

query_text = 'what is computer interface'

query_vector = vectorizer.transform(['query_text']).toarray()

results = index.query(query_vector, top_k=10)

for i in results:
    print(i.id, i.score)