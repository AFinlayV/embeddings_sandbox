"""
This is a sandbox for me to learn about embedding text,
semantic search and how to use that to answer questions about text
the sample text is an article about Dungeons and dragons open license
The goal is to be able to ask questions about the text and get answers using GPT3

Right now it just returns the most similar sentence to the question
I want it to find the 3 most similar sentences to the question and then return the next sentence and previous sentence
I also want to be able to ask follow-up questions and get answers from gpt3 with those sentences as context

I will be using (stealing from) the following resources:
https://github.com/daveshap
https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
ChatGPT
https://www.youtube.com/watch?v=ocxq84ocYi0 - OpenAI's New GPT 3.5 Embedding Model for Semantic Search - James Briggs


"""

import openai
import os
import re
import json
import pinecone
import gpt_index
import datetime
import numpy as np
from numpy.linalg import norm

with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()
VERBOSE = True


def vprint(*args):
    if VERBOSE:
        print(*args)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


file = open_file("text_examples/dnd_article.txt")
vprint(file)
#split file into a list of sentences.
sentence_list = re.split(r' *[\.\?!][\'"\)\]]* *', file)
vector_list = []

if not os.path.exists("embeddings/vector_list.txt"):
    for sentence in sentence_list:
        vector = gpt3_embedding(sentence)
        vector_list.append((vector, sentence))
        vprint(sentence, '\n\n', vector, '\n\n')
#     save_file("embeddings/vector_list.txt", vector_list('|'))
# else:
#     vector_list = open_file("embeddings/vector_list.txt").split('|')
while True:
    question = input("Ask a question about the article: ")
    vector_sim_list = []
    if question:
        q_embedding = gpt3_embedding(question)
        vprint(q_embedding)
        for vector, sentence in vector_list:
            similar = similarity(vector, q_embedding)
            vector_sim_list.append((similar, sentence))
            vprint(f'{sentence} is {similar} similar to {question}')
        vector_sim_list.sort(reverse=True)
        print("The most similar sentences are: ")
        for sentence in vector_sim_list[:3]:
            print(sentence)
    else:
        break
