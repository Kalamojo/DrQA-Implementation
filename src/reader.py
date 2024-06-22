import csv
import numpy as np
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.tokens import Token, Doc
from collections import Counter
import warnings

class Embedder(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.vocab_path = vocab_path
        self.embed_path = embed_path
        self.glove_path = glove_path
        self.vocab, self.embeddings = self.__load_embedder()
        self.dimensions = self.embeddings[0].shape[0]
        print(self.dimensions)
    
    def embed(self, word: str) -> np.ndarray:
        try:
            ind = self.vocab.index(word)
            return self.embeddings[ind]
        except ValueError:
            return np.zeros(self.dimensions)

    def fine_tune(self, words: list[str], documents: list[str], max_iterations: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        new_words = [token for token in words if token not in self.vocab]
        print(len(new_words))
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
        print("loading spacy")
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        print("done loading")

    def construct_vectors(self, documents: list[str], query: str) -> list[np.ndarray]:
        matrix_list = []
        query_tokens = self.match_tokenize(query)
        ind = 0
        for doc in self.nlp.pipe(documents, batch_size=1000, n_process=2):
            print(doc[:100])
            matrix = []
            counter = Counter([token.text for token in doc])
            for token in doc:
                # ind += 1
                # if ind != 5:
                #     continue;
                #print(doc[i].text)
                embedding = self.embedder.embed(token.text)
                match_vec = self.exact_match(token, query_tokens)
                token_vec = self.token_feature(token, counter)
                matrix.append(np.concatenate((embedding, match_vec, token_vec)))
                #print(match_vec, token_vec)
                #return;
            matrix_list.append(np.array(matrix))
        return matrix_list
                

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        tokenizer = self.nlp.tokenizer
        doc = tokenizer(' '.join(documents))
        all_words = [token.text for token in doc]
        print("spacy done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)

    def match_tokenize(self, sentence: str) -> Doc:
        self.nlp.tokenizer
        return self.nlp(sentence)

    def exact_match(self, word: Token, question_tokens: Doc) -> np.ndarray:
        og = int(word.text in set(token.text for token in question_tokens))
        low = int(word.text.lower() in set(token.text.lower() for token in question_tokens))
        lem = int(word.lemma_ in set(token.lemma_ for token in question_tokens))
        return np.array([og, low, lem])
    
    def token_feature(self, word: Token, counter: Counter) -> np.ndarray:
        pos = self.nlp.meta['labels']['tagger'].index(word.tag_)
        if word.ent_type_ != "":
            ner = self.nlp.meta['labels']['ner'].index(word.ent_type_) + 1
        else:
            ner = 0
        termF = counter[word.text]
        return np.array([pos, ner, termF])
