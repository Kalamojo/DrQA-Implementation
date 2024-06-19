import csv
import numpy as np

class Embedder(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.vocab_path = vocab_path
        self.embed_path = embed_path
        self.glove_path = glove_path
        self.model = self.__load_embedder()
    
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
    def __init__(self, vocab_path: str = None, embed_path: str = None) -> None:
        self.embedder = self.__load_embedder(vocab_path, embed_path)

    def __embedding(self, word: str) -> np.ndarray:
        pass;

    
