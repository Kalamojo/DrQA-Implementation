from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from nltk import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
import joblib
import json

class Vectorizer(TfidfVectorizer):
    def __init__(self, bigram: bool = True, vectorizer_path: str = None) -> None:
        self.bigram = bigram
        self.vectorizer_path = vectorizer_path
        self.__load_vectorizer()

    def __tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)
    
    def save_vectorizer(self, path: str) -> None:
        joblib.dump(self.__getstate__(), path, compress=True)
    
    def __load_vectorizer(self) -> None:
        if self.vectorizer_path:
            self.__setstate__(joblib.load(self.vectorizer_path))
        else:
            ngram = (1,2) if self.bigram else (1,1)
            super().__init__(ngram_range=ngram, lowercase=True, strip_accents='unicode', tokenizer=self.__tokenizer)
    
    def _warn_for_unused_params(self) -> None:
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.tokenizer = self.__tokenizer

class Retriever(object):
    def __init__(self, vectorizer_path: str = None, matrix_path: str = None) -> None:
        self.vectorizer = Vectorizer(vectorizer_path=vectorizer_path)
        self.doc_matrix = self.__load_matrix(matrix_path)
        self.supported_squads = ['1.1']
    
    def train_squad(self, json_file: str, save: bool = True) -> None:
        with open(json_file, "r") as f:
            squad = json.load(f)
        
        version = squad['version']
        if version not in self.supported_squads:
            raise ValueError(f"Squad version {version} is not currently supported. The available options are:
                             \n{self.supported_squads}")
        
        docs = []
    
    def 
    
    def __load_matrix(self, path: str) -> sparse.spmatrix:
        if path:
            return sparse.load_npz(path)
        return None
    
    def __save_matrix(self, path: str) -> None:
        if self.doc_matrix is None:
            raise ValueError("Document Matrix is `None`. Please train on dataset first")
        else:
            sparse.save_npz(path, self.doc_matrix, compressed=True)
