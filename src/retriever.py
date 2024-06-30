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

    def get_squad_docs(self, query: str, squad_path: str, num_docs: int = 5) -> list[str]:
        with open(squad_path, "r") as f:
            squad = json.load(f)
        
        version = squad.get('version', None)
        if version not in self.supported_squads:
            raise ValueError(f"Squad version {version} is not currently supported. The available options are: \n{self.supported_squads}")
        
        top_docs = self.retrieve_docs(query, num_docs)
        return [self.extract_page(squad['data'][ind]) for ind in top_docs]

    def extract_page(self, squad_entry: dict) -> str:
        page = ' '.join(squad_entry['title'].split('_')) + '\n\n\n'
        for paragraph in squad_entry['paragraphs']:
            page += paragraph['context'] + '\n\n'
        return page
    
    def get_squad_qas(self, squad_path: str) -> tuple[list[str], list[tuple[str, int]], list[tuple[int, int]]]:
        with open(squad_path, "r") as f:
            squad = json.load(f)
        
        version = squad.get('version', None)
        if version not in self.supported_squads:
            raise ValueError(f"Squad version {version} is not currently supported. The available options are: \n{self.supported_squads}")
        
        page_ind = 0
        pages = []
        questions = []
        answers = []
        for entry in squad['data']:
            page = ' '.join(entry['title'].split('_')) + '\n\n\n'
            offset = len(page)
            for paragraph in entry['paragraphs']:
                for qa in paragraph['qas']:
                    questions.append((qa['question'], page_ind))
                    answers.append(set((offset+ans['answer_start'], offset+ans['answer_start']+len(ans['text'])) 
                                       for ans in qa['answers']))
                page += paragraph['context'] + '\n\n'
                offset += len(paragraph['context'] + '\n\n')
            pages.append(page)
            page_ind += 1
        
        return pages, questions, answers
    
    def retrieve_docs(self, query: str, num_docs: int = 5) -> np.ndarray:
        if self.doc_matrix is None:
            raise ValueError("Document Matrix is `None`. Please run train_squad or supply a matrix_path in the constructor first")
        
        query_vector = self.vectorizer.transform([query])
        scores = self.__comparison_scores(query_vector, self.doc_matrix, metric='cosine')
        return scores[0][:num_docs]

    def get_squad_accuracy(self, squad_path: str, num_docs: int = 5) -> int:
        with open(squad_path, "r") as f:
            squad = json.load(f)
        
        version = squad.get('version', None)
        if version not in self.supported_squads:
            raise ValueError(f"Squad version {version} is not currently supported. The available options are: \n{self.supported_squads}")
        
        if self.doc_matrix is None:
            raise ValueError("Document Matrix is `None`. Please run train_squad or supply a matrix_path in the constructor first")
        
        correct = 0
        questions = []
        titles = []
        for entry in squad['data']:
            for paragraph in entry['paragraphs']:
                for qa in paragraph['qas']:
                    questions.append(qa['question'])
                    titles.append(entry['title'])
        query_matrix = self.vectorizer.transform(questions)
        scores = self.__comparison_scores(query_matrix, self.doc_matrix, metric='cosine')
        
        for i in range(len(questions)):
            found = False
            for j in range(num_docs):
                ind = scores[i][j]
                if squad['data'][ind]['title'] == titles[i]:
                    found = True
                    break;
            if found:
                correct += 1
        
        return correct / len(questions)

    def __comparison_scores(self, matrix1: sparse.spmatrix, matrix2: sparse.spmatrix, metric: str) -> np.ndarray:
        # cosine similarity calculation
        if metric == 'cosine':
            scores = np.dot(matrix1, matrix2.T)/(norm(matrix1)*norm(matrix2))
            score_inds = np.argsort(-scores.toarray())
            return score_inds
        elif metric == 'dot':
            scores = np.dot(matrix2, matrix1.T)
            score_inds = np.argsort(-scores.T.toarray())
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
