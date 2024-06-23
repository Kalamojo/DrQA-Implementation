from retriever import Retriever
from reader import Reader, NLTKCustomTokenizer
import spacy
import json
from nltk import word_tokenize, TreebankWordTokenizer, NLTKWordTokenizer, string_span_tokenize, WordPunctTokenizer, sent_tokenize, PunktSentenceTokenizer
from itertools import chain

def fineTune(reader, retriever):
    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    docs = []
    for entry in squad['data']:
        docs.append(retriever.extract_page(entry))
    reader.fine_tune_embedder(docs, vocab_save="vocab_v1T.vocab", embed_save="embeddings_v1T.npy")

def main():
    #doc_reader = Reader(glove_path="./data/glove.840B.300d.txt")
    doc_reader = Reader(vocab_path="vocab_v1T.vocab", embed_path="embeddings_v1T.npy")
    doc_retriever = Retriever(vectorizer_path="vectorizer_v1.pkl", matrix_path="matrix_v1.npz")
    # #doc_reader.embedder.save_embedder("vocab_v1.vocab", "embeddings_v1.npy")
    #fineTune(doc_reader, doc_retriever)

    query = "What was Beyoncé's first major breakout album?"
    docs = doc_retriever.get_squad_docs(query, "./data/train-v1.1.json")
    matrix_list = doc_reader.construct_vectors(docs, query)
    print(len(matrix_list))
    print(matrix_list[0].shape)
    print(matrix_list[0])
    
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
