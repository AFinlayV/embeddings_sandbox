import openai
import json
import re

with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    openai.api_key = f.read().strip()


def text_to_embedding(text):
    response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)



def main():
    in_filename = "text_examples/music_history.txt"
    out_filename = "embeddings/music_history.json"
    text = load_text(in_filename)
    # split text into sentences and remove empty strings
    text_list = re.split(r' *[\.\?!][\'"\)\]]* *', text)
    print(text, '\n\n\n', text_list)
    embeddings = {}
    for i in text_list:
        if i == '':
            continue
        print(i)
        vector = text_to_embedding(i)
        embeddings[i] = vector
    save_json(out_filename, embeddings)

if __name__ == "__main__":
    main()
