"""
Question and answering with references from embeddings.
use semantic search to find relevant sentences and then use those sentences as context for gpt3

"""

import openai
import os
import re
import json


with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def check_similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity

