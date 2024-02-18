from openai import ChatCompletion
import os 
import openai
from dotenv import load_dotenv
from transformers import pipeline
import sys
from SideNet.side_net import SideNet


load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY

class E_I_S_A:
    def __init__(self, config={"api_key": OPEN_AI_API_KEY}) -> None:
        self.config = config

    def episodic_interaction(self, prompt):
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=OPEN_AI_API_KEY,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        return chat.choices[0].message.content

    def ENCM(input): 
        need = None
        word_c = len(input.split())
        pipe = pipeline("text-classification", model="krupper/text-complexity-classification")
        res = pipe(input)
        if res[0]["label"] == "special_language":
            need = True
            data = SideNet(input)
        else: 
            need = False

config = {"api_key": OPEN_AI_API_KEY}

episodic_separation_architecture = E_I_S_A(config)
input = "what is cognitive science?"


generated_response = episodic_separation_architecture.episodic_interaction(input)
print("Generated Response:", generated_response)