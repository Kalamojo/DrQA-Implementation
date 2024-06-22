import csv
import h5py
import numpy as np
from mittens import Mittens, GloVe
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
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
        ind = self.vocab.index(word)
        return self.embeddings[ind]

    def fine_tune(self, words: list[str], documents: list[str], max_iterations: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        # new_words = [token for token in words if token not in self.model.keys()]
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
            f.writelines([line + '\n' for line in self.vocab])
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

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)
        self.lemmatizer = WordNetLemmatizer()

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        all_words = ' '.join(documents).split()
        counter = Counter(all_words)
        common_words, counts = zip(counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, vocab_save, embed_save)

    def __match_tokenize(self, sentence: str) -> tuple(list[str], list[str], list[str]):
        sentence_og = word_tokenize(sentence)
        sentence_lower = [word.lower() for word in sentence_og]
        sentence_lemma = [self.lemmatizer.lemmatize()]

    def __exact_match(self, word: str, question_og: list[str], question_lower: list[str], question_lemma: list[str]) -> np.ndarray:
        og = int(word in question_og)
        low = int(word.lower() in question_lower)
        lem = int(self.lemmatizer.lemmatize(word) in question_lemma)
        return np.array([og, low, lem])
