import csv
import numpy as np
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
import warnings

class Embedder(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.vocab_path = vocab_path
        self.embed_path = embed_path
        self.glove_path = glove_path
        self.model = self.__load_embedder()
        self.dimensions = list(self.model.values())[0].shape[0]
    
    def embed(self, word: str) -> np.ndarray:
        if word not in model:
            raise KeyError("Word has not been seen by the embedding model")
        return model[word]

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
        word_list, vectors = zip(*self.model.items())
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.writelines([line + '\n' for line in word_list])
        np.save(embed_path, vectors)

    def __load_embedder(self) -> dict[str: np.ndarray]:
        if self.vocab_path is not None and self.embed_path is not None:
            return self.__load_saved_embedder(self.vocab_path, self.embed_path)
        elif self.glove_path is not None:
            return self.__glove2dict(self.glove_path)
        
        raise ValueError("Either vocab and embed paths are required to load saved Embedder,\n or glove_path is required to create a new one")
    
    def __glove2dict(self, path) -> dict[str: np.ndarray]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            embeds = {line[0]: np.array(list(map(float, line[1:])))
                    for line in reader}
        return embeds
    
    def __load_saved_embedder(self, vocab_path: str, embed_path: str) -> dict[str: np.ndarray]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            word_list = [line.strip() for line in f]
        vectors = np.load(embed_path)
        model = {}
        for i, word in enumerate(word_list):
            model[word] = vectors[i]
        return model
        

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)

    def fine_tune_embedder(self, words: list[str], documents: list[str], vocab_save: str = None, embed_save: str = None) -> None:
        self.embedder.fine_tune(words, documents)

    
