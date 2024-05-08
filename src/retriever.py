from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import joblib

class Vectorizer1(object):
    def __init__(self, bigram: bool = True, vectorizer_path: str = None) -> None:
        self.vectorizer = self.__load_vectorizer(vectorizer_path, bigram)

    def fit_documents(self, documents: list[str]) -> None:
        self.vectorizer.fit(documents)

    def get_vectors(self, documents: list[str]) -> spmatrix:
        return self.vectorizer.transform(documents)

    def __tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)
    
    def __load_vectorizer(self, path: str = None, bigram: bool = True) -> TfidfVectorizer:
        if path:
            return joblib.load(path)
        else:
            ngram = (1,2) if bigram else (1,1)
            return TfidfVectorizer(ngram_range=ngram, lowercase=True, strip_accents='unicode', tokenizer=self.__tokenizer)
    
    def save_vectorizer(self, path: str) -> None:
        joblib.dump(self.vectorizer, path, compress=True)

class Vectorizer(TfidfVectorizer):
    def __new__(cls, *args, **kw):
        f = super().__new__(cls)
        f.__init__(*args, **kw)
        return f
    
    def __init__(self, bigram: bool = True, vectorizer_path: str = None) -> None:
        self.bigram = bigram
        self.vectorizer_path = vectorizer_path
        self.__load_vectorizer()

    def __tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)

    def __load_vectorizer(self) -> None:
        if self.vectorizer_path:
            thing = joblib.load(self.vectorizer_path)
            print(thing)
            self.__dict__ = thing
        else:
            ngram = (1,2) if self.bigram else (1,1)
            super().__init__(ngram_range=ngram, lowercase=True, strip_accents='unicode', tokenizer=self.__tokenizer)
    
    def save_vectorizer(self, path: str) -> None:
        joblib.dump(self.__dict__, path, compress=True)
