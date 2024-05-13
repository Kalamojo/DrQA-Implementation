from retriever import Retriever
import json

def main():
    #doc_retriever = Retriever()
    #doc_retriever.train_squad(squad_path="./data/train-v2.0.json", vectorizer_save="vectorizer.pkl", matrix_save="matrix.npz")
    doc_retriever = Retriever(vectorizer_path="vectorizer.pkl", matrix_path="matrix.npz")
    query = "Who was the first president of the United States?"
    top_docs = doc_retriever.retrieve_docs(query)

    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    
    print('\n'.join([squad['data'][top_docs[i]]['title'] for i in range(5)]))
    #print(doc_retriever.extract_page(squad['data'][top_docs[1]]))

if __name__ == '__main__':
    main()
