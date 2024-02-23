from openai import ChatCompletion
import os 
import openai
from dotenv import load_dotenv
from transformers import pipeline
import sys
from gensim.test.utils import common_texts, common_dictionary, common_corpus
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
from word_vecs.word_vecs import model


load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY
# model = Word2Vec(common_texts, min_count=1)
# vocab = set(model.wv.index_to_key)
class E_I_S_A:
    def __init__(self, config={"api_key": OPEN_AI_API_KEY}) -> None:
        self.config = config


    def episodic_interaction(self, prompt):
        search_result = self.ENCM(prompt)
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}, 
        ]
        if search_result is not None: 
            messages.append({"role": "system", "content": search_result})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=OPEN_AI_API_KEY,
            messages=messages
        )

        return chat.choices[0].message.content

    def ENCM(self, input): 

        spl = input.split()
        filt_mem = [word for word in spl if word in vocab]

        def SideNet():  
            db = []
            db.extend([model.wv['computer'], model.wv['interface']]) # FIXME: Add a route to acess the prompts instead of the extension
            data = []
            for word in filt_mem:
                vec = model.wv[word]
                data.append([word] + vec.tolist())  
            def Search(test_episodes): 
                # sim = []
                final_search = []
                # for db_entry in db: 
                #     for data_entry in data: 
                #         sim.append([abs(db_val - data_val) for db_val, data_val in zip(db_entry[1:], data_entry[1:])])
                # for k in sim: 
                #     if any(diff < 0.00005 for diff in k): 
                #         final_search.append(k)
                # if final_search:
                average_vector = np.mean(final_search, axis=0)
                most_similar_episode = None
                max_similarity_score = -1

                for episode in test_episodes:
                    spl_1 = episode.split()
                    filt_epi = [word for word in spl_1 if word in vocab]
                    episode_vector = np.mean([model.wv[word] for word in filt_epi], axis=0)

                if not np.isnan(episode_vector).any():
                            
                    similarity_score = cosine_similarity([episode_vector], [average_vector])[0][0]

                    if similarity_score > max_similarity_score:
                        most_similar_episode = episode
                        max_similarity_score = similarity_score

                    return most_similar_episode
                return None
                # else: 
                #     return None
            return Search(test_episodes)
            
        need = None
        word_c = len(input.split())
        pipe = pipeline("text-classification", model="krupper/text-complexity-classification")
        res = pipe(input)
        if res[0]["label"] == "special_language":
            need = True
        else: 
            need = False
        if need: 
            SideNet()

  
        

config = {"api_key": OPEN_AI_API_KEY}

episodic_separation_architecture = E_I_S_A(config)
input = "what exactly is a human being?"
test_episodes = ["Is a human a prompter?", "Is a dog a listener?", "Is a cat a speaker?"]

generated_response = episodic_separation_architecture.episodic_interaction(input)
print("Generated Response:", generated_response)

