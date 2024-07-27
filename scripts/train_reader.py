from drqa_imp import reader
import csv

with open("./data/sizes.csv") as f:
    csv_reader = csv.reader(f, delimiter=',')
    _ = next(csv_reader)
    line = next(csv_reader)
    max_question_size = int(line[0])
    max_paragraph_size = int(line[1])
    num_questions = int(line[2])

doc_reader = reader.Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy", max_q=max_question_size, max_p=max_paragraph_size)

doc_reader.train_reader(data_dir="./data/", count=num_questions, checkpoint_dir="models")
