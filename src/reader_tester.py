from retriever import Retriever
from reader import Reader, NLTKCustomTokenizer
import spacy
import json
from nltk import word_tokenize, TreebankWordTokenizer, NLTKWordTokenizer, string_span_tokenize, WordPunctTokenizer, sent_tokenize, PunktSentenceTokenizer
from itertools import chain
import csv
# from aligner import Aligner, AlignmentLayer, Attention
# import numpy as np

def fineTune(reader, retriever):
    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    docs = []
    for entry in squad['data']:
        docs.append(retriever.extract_page(entry))
    reader.fine_tune_embedder(docs, vocab_save="vocab_v1T.vocab", embed_save="embeddings_v1T.npy")

def main():
    # doc_reader = Reader(glove_path="./data/glove.840B.300d.txt")
    doc_reader = Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy", max_q=38, max_p=17200)
    doc_retriever = Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")
    # #doc_reader.embedder.save_embedder("vocab_v1.vocab", "embeddings_v1.npy")
    # fineTune(doc_reader, doc_retriever)

    # s = "howdy folks"
    # spans = list(TreebankWordTokenizer().span_tokenize(s))
    # print(spans)
    # print(s[spans[1][0]:spans[1][1]])

    query = "Who is Beyoncé married to?"
    docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json")
    doc_reader.test_reader(docs, query, 38, 17200, checkpoint_dir="models/")

    #ho sang a version of Queen's Somebody to Love in 2004's Ella Enchanted?
    #("What year did the government of Zhejiang recognise folk religion as 'civil religion'?", 97)
    #During which centuries did ROme fall under the influence of Byzantine art?
    #Demeter
    #How many copies of each chromosome does a sexual organism have?
    #What does Margulis think is the main driver of evolution?

    # #pages, questions, answers = doc_retriever.get_squad_qas("./data/train-v1.1.json")
    # doc_reader.train_reader(data_dir="./data/", count=1000, checkpoint_dir="models")
    # #doc_reader.test_train()

    # pages, questions, answers = doc_retriever.get_squad_qas("./data/train-v1.1.json")
    # doc_reader.prepare_training_data(pages, questions, answers, save_dir="./data/")
    # #doc_reader.test_train()

    # aligner = Aligner(embed_dim=300)
    # aligner.q_encoder.summary()
    # aligner.q_aligner.summary()

    # query_embedding = np.random.rand(59, 300, 1)
    # p_matrix = np.random.rand(1021, 300, 1)
    # aligned_vec = aligner.q_aligner([query_embedding, p_matrix], training=False)
    # print(aligned_vec)
    # print(aligned_vec.shape)

    # ind = 4365
    # print(questions[ind])
    # print(answers[ind])
    # ans = next(iter(answers[ind]))
    # print("!!!" + pages[questions[ind][1]][ans[0]:ans[1]] + "!!!")

    # print()
    # ind = 34734
    # print(questions[ind])
    # print(answers[ind])
    # ans = next(iter(answers[ind]))
    # print("!!!" + pages[questions[ind][1]][ans[0]:ans[1]] + "!!!")

    # query = "What was Beyoncé's first major breakout album?"
    # docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json")
    # matrix_list = doc_reader.construct_vectors(docs, query, False)
    # print(len(matrix_list))
    # print(matrix_list[0].shape)
    # print(matrix_list[0])
    
    # s = '''Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'''
    # big = len(s)
    # print([item for item in chain.from_iterable(((start+sent_start, end+sent_start) for start, end in TreebankWordTokenizer().span_tokenize(s[sent_start:sent_end])) for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(s))])
    # print([s[st:en] for st, en in chain.from_iterable(((start+sent_start, end+sent_start) for start, end in TreebankWordTokenizer().span_tokenize(s[sent_start:sent_end])) for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(s))])
    # print([s[en] == ' ' if en < big else False for _, en in chain.from_iterable(((start+sent_start, end+sent_start) for start, end in TreebankWordTokenizer().span_tokenize(s[sent_start:sent_end])) for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(s))])
    # spaces = []
    # words = []
    # for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(s):
    #     for start, end in TreebankWordTokenizer().span_tokenize(s[sent_start:sent_end]):
    #         words.append(s[start+sent_start:end+sent_start])
    #         if end+sent_start < sent_end and s[end+sent_start] == ' ':
    #             spaces.append(True)
    #         else:
    #             spaces.append(False)
    # print(words)
    # print(spaces)

    # nlp = spacy.load("en_core_web_sm", exclude="tokenizer")
    # #nlp = spacy.blank("en")
    # nlp.tokenizer = NLTKCustomTokenizer(nlp.vocab)
    # doc = nlp("How do you do on this fine day, Mr. Caleb sir?")
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #             token.shape_, token.is_alpha, token.is_stop)

if __name__ == '__main__':
    main()
