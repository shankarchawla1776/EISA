import openai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPEN_AI_API_KEY")

client = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "what is cognitive science?"}],
    api_key=API_KEY,
    stream=False
)

print(client.choices[0].message.content)

