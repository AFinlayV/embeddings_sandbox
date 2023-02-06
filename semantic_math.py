"""
I want this code to be able to take a string of text and return a vector of numbers
then I want to be able to perform mathematical operations on those vectors
then I want to be able to take a vector of numbers and return a string of text

The purpose of this would be operatioins like:
'king' - 'man' + woman = 'queen'

from ChatGPT:
import gensim.downloader as api

# Download a pre-trained word2vec model
word2vec_model = api.load("word2vec-google-news-300")

# Perform the king - man + woman = queen analogy
result = word2vec_model.most_similar(positive=['woman', 'king'], negative=['man'])

# Print the result
print("The closest match to 'king - man + woman' is:", result[0][0])

================================

gensim didn't work for me, so I'm going to try this:
import spacy
import numpy as np

# Load the medium-sized English model
nlp = spacy.load("en_core_web_md")

# Define the words you want to compare
word1 = nlp("king")
word2 = nlp("man")
word3 = nlp("woman")

# Perform vector addition to find the vector for "queen"
result = word1.vector - word2.vector + word3.vector

# Add an extra dimension to the input array
result = np.reshape(result, (1, result.shape[0]))

# Find the closest word to the result
most_similar = nlp.vocab.vectors.most_similar(result)

# Print the text representation of the closest word
print("Result:", nlp.vocab[most_similar[0][0]].text)
================================

This produces rusults but they are not very good. kinda nonsense words that are similar to the inputs:
the above code produces: kingi
"""

import openai
import os
import io
import json
import re
import numpy as np
from numpy.linalg import norm
import datetime
import gensim.downloader as api
import spacy
import numpy as np

# Load the medium-sized English model
nlp = spacy.load("en_core_web_md")

# Define the words you want to compare
word1 = nlp("king")
word2 = nlp("man")
word3 = nlp("woman")

# Perform vector addition to find the vector for "queen"
result = word1.vector - word2.vector + word3.vector

# Add an extra dimension to the input array
result = np.reshape(result, (1, result.shape[0]))

# Find the closest word to the result
most_similar = nlp.vocab.vectors.most_similar(result)

# Print the text representation of the closest word
print("Result:", nlp.vocab[most_similar[0][0]].text)

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


def vector_math(v1, v2, operation):
    if operation == '+':
        return np.add(v1, v2)
    elif operation == '-':
        return np.subtract(v1, v2)
    elif operation == '*':
        return np.multiply(v1, v2)
    elif operation == '/':
        return np.divide(v1, v2)
    elif operation == 'dot':
        return np.dot(v1, v2)
    elif operation == 'avg':
        return np.average([v1, v2], axis=0)
    else:
        raise ValueError("Operation must be one of '+', '-', '*', '/'")


def vector_to_text(vector):
    # TODO figure this out...
    # convert vector to text using gpt's api?? is this even possible? looking into other embedding models that can go both ways
    text = ''

