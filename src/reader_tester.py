from retriever import Retriever
from reader import Reader
import json

def fineTune(reader, retriever):
    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    docs = []
    for entry in squad['data']:
        docs.append(retriever.extract_page(entry))
    reader.fine_tune_embedder(docs, vocab_save="vocab_v1T.vocab", embed_save="embeddings_v1T.npy")

def main():
    # #doc_reader = Reader(glove_path="./data/glove.840B.300d.txt")
    doc_reader = Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy")
    doc_retriever = Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")
    # #doc_reader.embedder.save_embedder("vocab_v1.vocab", "embeddings_v1.npy")
    #fineTune(doc_reader, doc_retriever)

    query = "Who was the first president of the United States?"
    docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json")
    matrix_list = doc_reader.construct_vectors(docs, query)
    print(len(matrix_list))
    print(matrix_list[0].shape)

if __name__ == '__main__':
    main()
