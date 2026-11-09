import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# models = openai.Model.list()

# for m in models["data"]:
#     print(m["id"])

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    print(m.name)