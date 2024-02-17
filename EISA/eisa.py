# from LLM.llm import openaiGTPT3
from openai import ChatCompletion
import os 
import openai
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY

class E_I_S_A: 

    def __init__(self, config =  {"api_key": OPEN_AI_API_KEY}) -> None: 
        self.config = config 
        self.chat = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            api_key=config[OPEN_AI_API_KEY], 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ])

    def episodic_interaction(self, prompt, response): 
        pair = f"{prompt}\nUser: {response}"
        chat_response = self.chat.complete(pair)

        self.chat.append_interaction(pair, chat_response)

config = {OPEN_AI_API_KEY: "your_api_key_here"}

episodic_architecture = E_I_S_A(config)
prompt = "Hello, how are you?"

response = "I'm doing well, thank you. How about you?"
generated_response = episodic_architecture.episodic_interaction(prompt, response)
print("Generated Response:", generated_response)