
from LLM.llm import openaiGTPT3

class EpisodicInteractionSeperationArchitecure: 

    def __init__(self, config) -> None: 
        self.config = config 
        self.llm_agent = openaiGTPT3(config["openai_api_key"])

    def episodic_interaction(self, prompt, response): 
        self.llm_agent.interactions.append((prompt, response))

        