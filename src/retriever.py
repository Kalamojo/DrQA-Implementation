from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
import joblib

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
    
    def _warn_for_unused_params(self):
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

