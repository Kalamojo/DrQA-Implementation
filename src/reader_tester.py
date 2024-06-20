from retriever import Retriever
from reader import Reader

def main():
    #doc_reader = Reader(glove_path="./data/glove.840B.300d.txt")
    doc_reader = Reader(vocab_path="vocab_v1.vocab", embed_path="embeddings_v1.npy")
    doc_retriever = Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")
    #doc_reader.save_embedder("vocab_v1.vocab", "embeddings_v1.npy")

    query = "Who was the first president of the United States?"
    docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json")
    doc_reader.fine_tune_embedder(["who", "what", "when", "where", "why", "how"], docs)

if __name__ == '__main__':
    main()
