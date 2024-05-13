from retriever import Retriever
import json

def main():
    #doc_retriever = Retriever()
    #doc_retriever.train_squad(squad_path="./data/train-v2.0.json", vectorizer_save="vectorizer.pkl", matrix_save="matrix.npz")
    doc_retriever = Retriever(vectorizer_path="vectorizer.pkl", matrix_path="matrix.npz")
    query = "The roman empire"
    top_docs = doc_retriever.retrieve_docs(query, squad_path="./data/train-v2.0.json")

    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    
    ind = top_docs[0]
    print(doc_retriever.extract_page(squad['data'][ind])[:100])

if __name__ == '__main__':
    main()
