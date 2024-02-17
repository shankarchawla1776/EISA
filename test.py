import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(
    api_key = API_KEY,
)

response = client.chat.completions.create(
    model="gp[t-3.5-turbo", 
    messages=[{"role": "user", "content":"what is cognitive science?"}],
    stream = False
)

print(response.choices[0].message.content)
print(openai.VERSION)