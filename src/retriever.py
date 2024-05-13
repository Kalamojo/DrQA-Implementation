from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np
#from numpy.linalg import norm
from scipy.sparse.linalg import norm
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt', quiet=True)
import joblib
import json
import warnings

class Vectorizer(TfidfVectorizer):
    def __init__(self, bigram: bool = True, vectorizer_path: str = None) -> None:
        self.bigram = bigram
        self.vectorizer_path = vectorizer_path
        self.__load_vectorizer()
        self.stop_words = set(stopwords.words('english'))

    def __tokenizer(self, text: str) -> list[str]:
        word_tokens = word_tokenize(text)
        return [token for token in word_tokens if token.lower() not in self.stop_words]
    
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
        self.supported_squads = ['1.1', 'v2.0']
        self.metric_options = ['dot', 'cosine']
    
    def train_squad(self, squad_path: str, vectorizer_save: str = None, matrix_save: str = None) -> None:
        with open(squad_path, "r") as f:
            squad = json.load(f)
        
        version = squad.get('version', None)
        if version not in self.supported_squads:
            raise ValueError(f"Squad version {version} is not currently supported. The available options are: \n{self.supported_squads}")
        
        docs = []
        for entry in squad['data']:
            docs.append(self.extract_page(entry))
        self.doc_matrix = self.vectorizer.fit_transform(docs)

        if matrix_save is not None and vectorizer_save is not None:
            self.__save_matrix(matrix_save)
            self.vectorizer.save_vectorizer(vectorizer_save)
        else:
            warnings.warn("Vectorizer and Matrix were not saved because paths were not supplied")
    
    def extract_page(self, squad_entry: dict) -> str:
        page = squad_entry['title'] + '\n\n\n'
        for paragraph in squad_entry['paragraphs']:
            page += paragraph['context'] + '\n\n'
        return page
    
    def retrieve_docs(self, query: str, squad_path: str, num_docs: int = 5) -> list[str]:
        if self.doc_matrix is None:
            raise ValueError("Document Matrix is `None`. Please run train_squad or supply a matrix_path in the constructor first")
        
        query_vector = self.vectorizer.transform([query])
        scores = self.__comparison_scores(query_vector, self.doc_matrix, metric='cosine')
        return scores[:num_docs]

    def __comparison_scores(self, vector: sparse.spmatrix, matrix: sparse.spmatrix, metric: str) -> np.ndarray:
        # cosine similarity calculation
        # print(vector.shape)
        # print(matrix.shape)
        if metric == 'cosine':
            scores = np.dot(vector, matrix.T)/(norm(vector)*norm(matrix))
            #score_inds = np.argsort(-scores.T.toarray())[0]
            score_inds = np.argsort(-scores.toarray())[0]
            print(scores)
            print(score_inds)
            print(score_inds.shape)
            print(scores.toarray()[score_inds[0]])
            return score_inds
        elif metric == 'dot':
            scores = np.dot(vector, matrix.T)
            score_inds = np.argsort(-scores.toarray())[0]
            # print(scores)
            # print(score_inds)
            # print(score_inds.shape)
            return score_inds
        else:
            raise ValueError(f"Metric {metric} not an option. Options:\n{self.metric_options}")
    
    def __load_matrix(self, path: str) -> sparse.spmatrix:
        if path:
            return sparse.load_npz(path)
        return None
    
    def __save_matrix(self, path: str) -> None:
        if self.doc_matrix is None:
            raise ValueError("Document Matrix is `None`. Please train on dataset first")
        else:
            sparse.save_npz(path, self.doc_matrix, compressed=True)
