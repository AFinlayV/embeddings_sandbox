import spacy
import datasets
import openai
import json
import datetime
import os
import numpy as np
from numpy.linalg import norm

from torch import cosine_similarity

with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()

VERBOSE = True


def vprint(*args):
    if VERBOSE:
        print(*args)


# Utility functions
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


def load_text(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text


# Pre-processing function
def preprocess(text):
    # Load the text into the spaCy NLP pipeline
    doc = nlp(text)

    # Create a list of lowercase tokens, filtering out stop words and punctuation
    text = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Join the filtered tokens into a single string
    return " ".join(text)


# Generate embeddings for the training manual text

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


# Semantic search function
def semantic_search(query, embeddings, num_results=5):
    # Pre-process the query text
    #preprocessed_query = preprocess(query)

    # Load the pre-processed query text into the spaCy NLP pipeline
    query_doc = gpt3_embedding(query)

    # Generate word embeddings for each token in the query document
    query_embeddings = [token.vector for token in query_doc]

    # Calculate the average embedding for the query
    query_embedding = np.mean(query_embeddings, axis=0)

    # Calculate the semantic similarity between the query and each sentence in the training manual
    similarities = [similarity(query_embedding, embedding) for embedding in embeddings]

    # Sort the results by semantic similarity
    results = sorted(zip(range(len(similarities)), similarities), key=lambda x: x[1], reverse=True)

    # Return the top 5 results
    return results[:num_results]


def main():
    # Load the training manual text
    data = load_json('/Users/alexthe5th/PycharmProjects/Data_Sets/us_history.json')
    text = data['text']
    # Embed each sentence in the training manual
    # make a list of strings for each sentence
    sentences = list(text.split('.'))
    vprint(sentences)
    # embed each sentence
    # if embedding file doesn't exist, create it
    if not os.path.exists('/Users/alexthe5th/PycharmProjects/Data_Sets/us_history_embeddings.json'):
        embeddings = {}
        for sentence in sentences:
            embeddings[sentence] = gpt3_embedding(sentence)
        # save the embeddings in json format
        save_json('/Users/alexthe5th/PycharmProjects/Data_Sets/us_history_embeddings.json', embeddings)
    else:
        embeddings = load_json('/Users/alexthe5th/PycharmProjects/Data_Sets/us_history_embeddings.json')
    # get user query
    query = input('Enter a query: ')
    # search for query
    results = semantic_search(query, embeddings)
    print(results)


if __name__ == '__main__':
    main()
