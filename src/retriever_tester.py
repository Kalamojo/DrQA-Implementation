from retriever import Retriever
import json

def main():
    # doc_retriever = Retriever()
    # doc_retriever.train_squad(squad_path="./data/train-v1.1.json", vectorizer_save="vectorizer_v1.pkl", matrix_save="matrix_v1.npz")
    doc_retriever = Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")
    query = "Who led the American Revolution to victory against Great Britain?"
    top_docs = doc_retriever.retrieve_docs(query)

    print(top_docs)
    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    
    print('\n'.join([squad['data'][top_docs[i]]['title'] for i in range(5)]))
    #print(doc_retriever.extract_page(squad['data'][top_docs[1]]))
    # acc = doc_retriever.get_squad_accuracy("./data/train-v1.1.json", num_docs=5)
    # print(acc)

if __name__ == '__main__':
    main()
