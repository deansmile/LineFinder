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
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)

models = client.models.list()

for m in models.data:
    print(m.id)