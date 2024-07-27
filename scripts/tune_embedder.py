from drqa_imp import reader
import json

def extract_page(squad_entry: dict) -> str:
    page = ' '.join(squad_entry['title'].split('_')) + '\n\n\n'
    for paragraph in squad_entry['paragraphs']:
        page += paragraph['context'] + '\n\n'
    return page

doc_reader = reader.Reader(glove_path="./data/glove.840B.300d.txt")

with open("./data/train-v1.1.json", "r") as f:
    squad = json.load(f)
docs = []
for entry in squad['data']:
    docs.append(extract_page(entry))

doc_reader.fine_tune_embedder(docs, vocab_save="vocab_v1T.vocab", embed_save="embeddings_v1T.npy")
