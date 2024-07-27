import sys
from drqa_imp import retriever
import json

num_docs = int(sys.argv[1])
query = ' '.join(sys.argv[2:])
print(query)

doc_retriever = retriever.Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")

top_docs = doc_retriever.retrieve_docs(query, num_docs=num_docs)

with open("./data/train-v1.1.json", "r") as f:
    squad = json.load(f)
print(top_docs)
print('\n'.join([squad['data'][top_docs[i]]['title'] for i in range(num_docs)]))
