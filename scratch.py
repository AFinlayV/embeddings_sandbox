import openai
import json
import numpy as np
from numpy.linalg import norm

with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()


def text_to_embedding(text):
    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
    return response



def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


text1 = "The dog jumps under the foxes"
text2 = "there is a candle on the table"
emb1 = text_to_embedding(text1)['data'][0]['embedding']
emb2 = text_to_embedding(text2)['data'][0]['embedding']
sim = similarity(emb1, emb2)
print(sim)

