from drqa_imp import retriever, reader

doc_reader = reader.Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy")
doc_retriever = retriever.Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")

pages, questions, answers = doc_retriever.get_squad_qas("./data/train-v1.1.json")
doc_reader.prepare_training_data(pages, questions, answers, save_dir="./data/")
