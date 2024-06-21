import csv
import h5py
import numpy as np
from mittens import Mittens, GloVe
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
import time

class Embedder(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.vocab_path = vocab_path
        self.embed_path = embed_path
        self.glove_path = glove_path
        self.vocab, self.embeddings = self.__load_embedder()
        self.dimensions = self.embeddings.shape[1]
        print(self.dimensions)
    
    def embed(self, word: str) -> np.ndarray:
        if word not in self.model:
            raise KeyError("Word has not been seen by the embedding model")
        return self.model[word]

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

        mittens_model = Mittens(n=self.dimensions, max_iter=max_iterations)
        new_embeddings = mittens_model.fit(
            coocc_matrix,
            vocab=corp_vocab,
            initial_embedding_dict=self.model)

        for i in range(len(corp_vocab)):
            self.model[corp_vocab[i]] = new_embeddings[i, :]

        if vocab_save is not None and embed_save is not None:
            self.save_embedder(vocab_save, embed_save)
        else:
            warnings.warn("Vectorizer and Matrix were not saved because paths were not supplied")
    
    def save_embedder(self, vocab_path: str, embed_path: str) -> None:
        #word_list, vectors = zip(*self.model.items())
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.writelines([line + '\n' for line in self.vocab])
        np.save(embed_path, self.embeddings)

    def __load_embedder(self) -> tuple[list[str], np.ndarray]:
        if self.vocab_path is not None and self.embed_path is not None:
            return self.__load_saved_embedder(self.vocab_path, self.embed_path)
        elif self.glove_path is not None:
            return self.__glove2dict(self.glove_path)
        
        raise ValueError("Either vocab and embed paths are required to load saved Embedder,\n or glove_path is required to create a new one")
    
    def __glove2dict(self, path) -> tuple[list[str], np.ndarray]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            vocab = []
            embeddings = []
            for line in reader:
                vocab.append(line[0])
                embeddings.append(np.array(list(map(float, line[1:]))))
        return vocab, np.array(embeddings)
    
    def __load_saved_embedder(self, vocab_path: str, embed_path: str) -> tuple[list[str], np.ndarray]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        embeddings = np.load(embed_path, mmap_mode='r')
        return vocab, embeddings

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        all_words = ' '.join(documents).split()
        counter = Counter(all_words)
        common_words, counts = zip(counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, vocab_save, embed_save)

    
