

import openai 

class openaiGTPT3: 
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def query(self, prompt, max_tokens=50, temperature=0.7, engine="davinci"): 
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].text.strip()
    
