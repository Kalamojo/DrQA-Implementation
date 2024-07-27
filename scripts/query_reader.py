import sys
from drqa_imp import retriever, reader
import csv

num_docs = int(sys.argv[1])
query = ' '.join(sys.argv[2:])
print(query)

with open("./data/sizes.csv") as f:
    csv_reader = csv.reader(f, delimiter=',')
    _ = next(csv_reader)
    line = next(csv_reader)
    max_question_size = int(line[0])
    max_paragraph_size = int(line[1])

doc_reader = reader.Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy", max_q=max_question_size, max_p=max_paragraph_size)
doc_retriever = retriever.Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")

docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json", num_docs=num_docs)
doc_reader.test_reader(docs, query, max_question_size, max_paragraph_size, checkpoint_dir="models/")
