from aligner import Aligner
from retriever import Retriever
import csv
import numpy as np
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
from nltk import TreebankWordTokenizer, PunktSentenceTokenizer
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Token, Doc
from collections import Counter
from itertools import chain
import warnings
import time

class Embedder(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.vocab_path = vocab_path
        self.embed_path = embed_path
        self.glove_path = glove_path
        self.vocab, self.embeddings = self.__load_embedder()
        self.dimensions = self.embeddings[0].shape[0]
        print(self.dimensions)
    
    def embed(self, word: str) -> np.ndarray:
        try:
            ind = self.vocab.index(word)
            return self.embeddings[ind]
        except ValueError:
            #print(word)
            return np.zeros(self.dimensions)

    def fine_tune(self, words: list[str], documents: list[str], max_iterations: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        # new_words = [token for token in words if token not in self.vocab]
        # print(len(new_words))
        corp_vocab = list(set(words))
        big_doc = [' '.join(documents)]
        vectorizer = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
        X = vectorizer.fit_transform(big_doc)
        Xc = (X.T * X)
        Xc.setdiag(0)
        coocc_matrix = Xc.toarray()
        print(coocc_matrix.shape)

        pre_trained = dict(zip(self.vocab, self.embeddings))
        mittens_model = Mittens(n=self.dimensions, max_iter=max_iterations)
        new_embeddings = mittens_model.fit(
            coocc_matrix,
            vocab=corp_vocab,
            initial_embedding_dict=pre_trained)

        for i in range(len(corp_vocab)):
            pre_trained[corp_vocab[i]] = new_embeddings[i, :]

        self.vocab, self.embeddings = zip(*pre_trained.items())

        if vocab_save is not None and embed_save is not None:
            self.save_embedder(vocab_save, embed_save)
        else:
            warnings.warn("Vectorizer and Matrix were not saved because paths were not supplied")
    
    def save_embedder(self, vocab_path: str, embed_path: str) -> None:
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.writelines((line + '\n' for line in self.vocab))
        np.save(embed_path, self.embeddings)

    def __load_embedder(self) -> tuple[list[str], list[np.ndarray]]:
        if self.vocab_path is not None and self.embed_path is not None:
            return self.__load_saved_embedder(self.vocab_path, self.embed_path)
        elif self.glove_path is not None:
            return self.__glove2dict(self.glove_path)
        
        raise ValueError("Either vocab and embed paths are required to load saved Embedder,\n or glove_path is required to create a new one")
    
    def __glove2dict(self, path) -> tuple[list[str], list[np.ndarray]]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            vocab = []
            embeddings = []
            for line in reader:
                vocab.append(line[0])
                embeddings.append(np.array(list(map(float, line[1:]))))
        return vocab, embeddings
    
    def __load_saved_embedder(self, vocab_path: str, embed_path: str) -> tuple[list[str], list[np.ndarray]]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        embeddings = np.load(embed_path, mmap_mode='r')
        return vocab, embeddings

class NLTKCustomTokenizer(object):
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        spaces = []
        words = []
        for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(text):
            for start, end in TreebankWordTokenizer().span_tokenize(text[sent_start:sent_end]):
                words.append(text[start+sent_start:end+sent_start])
                if end+sent_start < sent_end and text[end+sent_start] == ' ':
                    spaces.append(True)
                else:
                    spaces.append(False)
        return Doc(self.vocab, words=words, spaces=spaces)

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None, question_pad: int = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)
        #spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm", exclude=['tok2vec'])
        self.nlp.tokenizer = NLTKCustomTokenizer(self.nlp.vocab)
        self.aligner = Aligner(self.embedder.dimensions, None)

    def train_reader(self, doc_retriever: Retriever, squad_path: str, documents: list[str], questions: list[tuple[str, int]], answers: list[tuple[int, int]], num_docs = 5, num_questions: int = 100):
        questions_list = [TreebankWordTokenizer().tokenize(question[0]) for question in questions]
        max_words = max(len(q_list) for q_list in questions_list)
        self.aligner.create_question_encoder(input_shape=(self.embedder.dimensions, 1))
        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        
        for i in range(len(questions)):
            doc = documents[questions[i][1]]
            title = doc[:doc.find('\n\n\n')]
            retrieved_docs = doc_retriever.get_squad_docs(questions[i][0], squad_path, num_docs)
            #print(retrieved_docs)
            titles = [d[:d.find('\n\n\n')] for d in retrieved_docs]
            if title not in titles:
                retrieved_docs.pop()
                retrieved_docs.append(doc)
            
            paragraph_vectors, query_vector = self.__construct_vectors(retrieved_docs, questions_list[i], max_words, False)
            print(paragraph_vectors.shape, query_vector.shape)
            return;

    def __construct_vectors(self, documents: list[str], query_list: list[str], pad_size: int, train = False) -> tuple[list[np.ndarray], np.ndarray]:
        matrix_list = []
        start = time.time()
        zeroes = np.zeros((pad_size - len(query_list), self.embedder.dimensions))
        query_embedding = np.concatenate((zeroes, np.array([self.embedder.embed(word) for word in query_list])))
        #print(query_embedding)
        print(query_embedding.shape)
        #return;
        query_ind_embedding = self.aligner.q_encoder(query_embedding.reshape((pad_size, self.embedder.dimensions, 1)), training=train)
        print(query_ind_embedding.shape)
        for doc in self.nlp.pipe(documents, batch_size=2, n_process=4):
            print(doc.text[:100])
            matrix = []
            #print("counting doc of length", len(doc.text))
            counter = Counter((token.text for token in doc))
            #print("done counting")
            for token in doc:
                #print("is it even entering")
                p_embedding = self.embedder.embed(token.text)
                #print("embedded")
                match_vec = self.__exact_match(token, query_list)
                #print("matched")
                token_vec = self.__token_feature(token, counter)
                # print("featured")
                # print(p_embedding.shape)
                # print((p_embedding.shape[0], p_embedding.shape[1], 1))
                p_embedding_shaped = p_embedding.reshape((p_embedding.shape[0], 1))
                print("starting aligning")
                aligned_vec = self.aligner.q_aligner([query_embedding.reshape(1, query_embedding[0], query_embedding[1]), p_embedding_shaped], training=train)
                print("done aligning")
                print(aligned_vec.shape)
                return;
                matrix.append(np.concatenate((aligned_vec, p_embedding, match_vec, token_vec)))
            matrix = np.array(matrix)
            matrix[:, -3:] = matrix[:, -3:]/np.linalg.norm(matrix[:, -3:], axis=0)[:, None]
            matrix_list.append(matrix)
        
        end = time.time()
        print("took", end - start, "seconds")
        return matrix_list, query_ind_embedding

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        joined_docs = ' '.join(documents)
        all_words = chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(joined_docs))
        print("nltk done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)

    def __exact_match(self, word: Token, question: list[str]) -> np.ndarray:
        og = int(word.text in question)
        low = int(word.text.lower() in question)
        lem = int(word.lemma_ in question)
        return np.array([og, low, lem])
    
    def __token_feature(self, word: Token, counter: Counter) -> np.ndarray:
        pos = self.nlp.meta['labels']['tagger'].index(word.tag_)
        if word.ent_type_ != "":
            ner = self.nlp.meta['labels']['ner'].index(word.ent_type_) + 1
        else:
            ner = 0
        termF = counter[word.text]
        return np.array([pos, ner, termF])
