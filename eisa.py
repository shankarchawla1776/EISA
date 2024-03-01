from openai import ChatCompletion
import os 
import openai
from dotenv import load_dotenv
from transformers import pipeline
from gensim.test.utils import common_texts, common_dictionary, common_corpus
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from components.mem.mem_arch import mem_arch
from database.word_vecs import model, vocab


load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY
# model = Word2Vec(common_texts, min_count=1)
# vocab = set(model.wv.index_to_key)
class E_I_S_A:
    def __init__(self, config={"api_key": OPEN_AI_API_KEY}) -> None:
        self.config = config


        
    def ENCM(self, input): 
        def SideNet(input_text): 
            word_vector_processor = mem_arch(input_text)
            word_vector_processor.process_word_vectors()
            return word_vector_processor.mem
        vectors = None
        pipe = pipeline("text-classification", model="krupper/text-complexity-classification")
        res = pipe(input)
        if res[0]["label"] == "special_language":
            need = True
        else: 
            need = False
        if need:
            vectors = self.SideNet(input)

    def LLM(self, input):
        search_result = self.ENCM(input)
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input}, 
        ]
        if search_result is not None: 
            messages.append({"role": "system", "content": search_result})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=OPEN_AI_API_KEY,
            messages=messages
        )

        return chat.choices[0].message.content


   

config = {"api_key": OPEN_AI_API_KEY}


# episodic_separation_architecture = E_I_S_A(config)

# input = "what exactly is a human being?"

generated_response = E_I_S_A.LLM("what exactly is a human being?")
print("Generated Response:", generated_response)
# generated_response = episodic_separation_architecture.LLM(input)
# print("Generated Response:", generated_response)

