from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import joblib

class DocumentRetriever:
    def __init__(self, bigram: bool = True, vectorizer_path: str = None) -> None:
        self.vectorizer = self.load_vectorizer(vectorizer_path, bigram)

    def fit_documents(self, documents: list[str]) -> None:
        self.vectorizer.fit(documents)

    def get_vectors(self, documents: list[str]) -> spmatrix:
        return self.vectorizer.transform(documents)

    def tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)
    
    def load_vectorizer(self, path: str = None, bigram: bool = True) -> TfidfVectorizer:
        if path:
            return joblib.load(path)
        else:
            ngram = (1,2) if bigram else (1,1)
            return TfidfVectorizer(ngram_range=ngram, lowercase=True, strip_accents='unicode', tokenizer=self.tokenizer)
    
    def save_vectorizer(self, path: str) -> None:
        joblib.dump(self.vectorizer, path, compress=True)
