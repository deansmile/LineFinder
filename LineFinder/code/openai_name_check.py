import os
try:  # load API keys from the repo-root .env (python-dotenv optional)
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass
# import openai


# models = openai.Model.list()

# for m in models["data"]:
#     print(m["id"])

# from google import genai


# for m in client.models.list():
#     print(m.name)

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)

models = client.models.list()

for m in models.data:
    print(m.id)