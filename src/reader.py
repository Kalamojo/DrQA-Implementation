from aligner import Aligner, AlignmentLayer, Attention
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
from joblib import Parallel, delayed
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

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
        self.nlp = spacy.load("en_core_web_sm", exclude=['parser'])
        self.nlp.tokenizer = NLTKCustomTokenizer(self.nlp.vocab)
        self.aligner = Aligner(self.embedder.dimensions)
        self.train = False

    def train_reader(self, doc_retriever: Retriever, squad_path: str, documents: list[str], questions: list[tuple[str, int]], answers: list[tuple[int, int]], num_docs = 5, num_questions: int = 100):
        questions_list = [TreebankWordTokenizer().tokenize(question[0]) for question in questions]
        max_words = max(len(q_list) for q_list in questions_list)
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
        
        zeroes = np.zeros((pad_size - len(query_list), self.embedder.dimensions))
        query_embedding = np.concatenate((zeroes, np.array([self.embedder.embed(word) for word in query_list]))).reshape((pad_size, self.embedder.dimensions, 1))
        #print(query_embedding)
        #print(query_embedding.shape)
        #return;
        query_ind_embedding = self.aligner.q_encoder(query_embedding, training=train)
        #print(query_ind_embedding)
        #print(query_ind_embedding.shape)

        joined_documents = '\n\n\n\n'.join(documents)
        all_words = chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(joined_documents))

        counter = Counter(all_words)

        all_words_copy = chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(joined_documents))
        p_embeddings = np.array([self.embedder.embed(word) for word in all_words_copy])
        print(p_embeddings.shape)
        n_in = p_embeddings.shape[0]
        input_dim = p_embeddings.shape[1]
        p_embeddings_shaped = p_embeddings.reshape((n_in, input_dim, 1))
        print(p_embeddings_shaped.shape)
        aligned_matrix = self.aligner.q_aligner([query_embedding, p_embeddings_shaped], training=self.train)
        print(aligned_matrix.shape)

        paragraphs = list(filter(None, joined_documents.split('\n')))
        
        #print(len(paragraphs))
        #count = 0
        start = time.time()
        # for doc in self.nlp.pipe(paragraphs, batch_size=2, n_process=4):
        #     matrix_list.append(self.__featurize(doc, query_list, counter))
        matrix_list = self.preprocess_parallel(paragraphs, len(paragraphs), set(query_list), counter)
        matrix_arr = np.vstack(matrix_list)
        print(matrix_arr.shape)
        matrix_arr = np.hstack([aligned_matrix, p_embeddings, matrix_arr])
        print(matrix_arr.shape)
        matrix_arr[:, -3:] = matrix_arr[:, -3:]/np.linalg.norm(matrix_arr[:, -3:], axis=1)[:, None]
        #matrix_list.append(matrix)
        
        end = time.time()
        print("took", end - start, "seconds")
        return matrix_arr, query_ind_embedding

    def __featurize(self, doc: Doc, query_set: set[str], counter: Counter) -> list[np.ndarray]:
        matrix = []
        for token in doc:
            match_vec = self.__exact_match(token, query_set)
            token_vec = self.__token_feature(token, counter[token.text])
            matrix.append(np.concatenate((match_vec, token_vec)))
        return matrix
    
    def chunker(self, iterable, total_length, chunksize):
        return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))
    
    def flatten(self, list_of_lists: list[list]) -> list:
        return [item for sublist in list_of_lists for item in sublist]
    
    def process_chunk(self, texts, query_set: set[str], counter: Counter) -> list[list[np.ndarray]]:
        matrix_list = []
        for doc in self.nlp.pipe(texts, batch_size=20):
            matrix_list.append(self.__featurize(doc, query_set, counter))
        return matrix_list
    
    def preprocess_parallel(self, texts, num_texts: int, query_set: set[str], counter: Counter, chunksize: int = 70, n_jobs: int = 7) -> list[np.ndarray]:
        executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer='processes', max_nbytes=None)
        do = delayed(self.process_chunk)
        tasks = (do(chunk, query_set, counter) for chunk in self.chunker(texts, num_texts, chunksize=chunksize))
        result = executor(tasks)
        return self.flatten(result)

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        joined_docs = ' '.join(documents)
        all_words = chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(joined_docs))
        print("nltk done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)

    def __exact_match(self, word: Token, question: set[str]) -> np.ndarray:
        og = int(word.text in question)
        low = int(word.text.lower() in question)
        lem = int(word.lemma_ in question)
        return np.array([og, low, lem])
    
    def __token_feature(self, word: Token, freq: int) -> np.ndarray:
        pos = self.nlp.meta['labels']['tagger'].index(word.tag_)
        if word.ent_type_ != "":
            ner = self.nlp.meta['labels']['ner'].index(word.ent_type_) + 1
        else:
            ner = 0
        return np.array([pos, ner, freq])
