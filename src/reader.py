import csv
import numpy as np
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
from nltk import TreebankWordTokenizer, PunktSentenceTokenizer
import spacy
import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Layer, SimpleRNN
import keras.backend as K
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

class AlignmentLayer(Layer):
    def __init__(self, **kwargs):
        super(AlignmentLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[1][-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1][1], 1),
            initializer="zeros", trainable=True)
        super(AlignmentLayer, self).build(input_shape)

    def call(self, x):
        embed_q, embed_p = x
        print(embed_q.shape, embed_p.shape)
        query_p = embed_p @ self.W + self.b
        # Alignment scores
        scores = K.tanh(K.dot(embed_q, query_p))
        # Remove dimension of size 1
        alpha = K.softmax(scores)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute context vector
        context = embed_q * alpha
        context = K.sum(context, axis=1)
        return context

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
            initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        scores = K.tanh(K.dot(x, self.W)+self.b)
        # Remove dimension of size 1
        alpha = K.softmax(scores)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)
        #spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm", exclude=['tok2vec'])
        self.nlp.tokenizer = NLTKCustomTokenizer(self.nlp.vocab)
        self.qp_aligner = self.create_question_aligner()
        self.ind_embedder = self.create_independant_embedder()

    def construct_vectors(self, documents: list[str], query: str, train=False) -> tuple[list[np.ndarray], np.ndarray]:
        matrix_list = []
        start = time.time()
        query_tokens = self.__match_tokenize(query)
        query_embedding = np.array([self.embedder.embed(token.text) for token in query_tokens])
        query_ind_embedding = self.ind_embedder(query_embedding, training=train)
        print(query_embedding.shape)
        for doc in self.nlp.pipe(documents, batch_size=2, n_process=4):
            #print(doc.text[:100])
            matrix = []
            counter = Counter((token.text for token in doc))
            for token in doc:
                p_embedding = self.embedder.embed(token.text)
                match_vec = self.__exact_match(token, query_tokens)
                token_vec = self.__token_feature(token, counter)
                aligned_vec = self.qp_aligner([query_embedding, p_embedding], training=train)
                matrix.append(np.concatenate((p_embedding, match_vec, token_vec, aligned_vec)))
            matrix = np.array(matrix)
            f_start, f_end = -1 * query_embedding.shape[0] - 3, -1 * query_embedding.shape[0]
            matrix[:, f_start:f_end] = matrix[:, f_start:f_end]/np.linalg.norm(matrix[:, f_start:f_end], axis=0)[:, None]
            matrix_list.append(matrix)
        
        end = time.time()
        print("took", end - start, "seconds")
        return matrix_list, query_ind_embedding

    def create_question_aligner(self, input_shape: list[tuple] = [(300,), (300,)]) -> Model:
        x_q = Input(shape=input_shape[0])
        x_p = Input(shape=input_shape[1])
        alignment_layer = AlignmentLayer()([x_q, x_p])
        model = Model([x_q, x_p], alignment_layer)
        assert model.output_shape == (input_shape[0][1], input_shape[0][2])
        return model

    def create_independant_embedder(self, input_shape: tuple = (300,), hidden_units: int = 20, dense_units: int = 20, activation: str = "tanh") -> Model:
        x = Input(shape=input_shape)
        RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
        attention_layer = Attention()(RNN_layer)
        model = Model(x, attention_layer)
        assert model.output_shape == (input_shape[0], input_shape[1])
        return model

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        joined_docs = ' '.join(documents)
        all_words = chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(joined_docs))
        print("nltk done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)

    def __match_tokenize(self, sentence: str) -> Doc:
        return self.nlp(sentence)

    def __exact_match(self, word: Token, question_tokens: Doc) -> np.ndarray:
        og = int(word.text in (token.text for token in question_tokens))
        low = int(word.text.lower() in (token.text.lower() for token in question_tokens))
        lem = int(word.lemma_ in (token.lemma_ for token in question_tokens))
        return np.array([og, low, lem])
    
    def __token_feature(self, word: Token, counter: Counter) -> np.ndarray:
        pos = self.nlp.meta['labels']['tagger'].index(word.tag_)
        if word.ent_type_ != "":
            ner = self.nlp.meta['labels']['ner'].index(word.ent_type_) + 1
        else:
            ner = 0
        termF = counter[word.text]
        return np.array([pos, ner, termF])
