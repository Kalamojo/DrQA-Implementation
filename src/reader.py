from aligner import Aligner
from retriever import Retriever
import csv
import numpy as np
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
from nltk import TreebankWordTokenizer, PunktSentenceTokenizer
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Token, Doc
from spacy.vectors import Vectors
from collections import Counter
from itertools import chain
import random
from collections.abc import Iterator, Generator
import warnings
import time
from joblib import Parallel, delayed
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
            #print(word)
            return np.zeros(self.dimensions)

    def fine_tune(self, words: list[str], documents: list[str], max_iterations: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        # new_words = [token for token in words if token not in self.vocab]
        # print(len(new_words))
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
            f.writelines((line + '\n' for line in self.vocab))
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

class NLTKCustomTokenizer(object):
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        spaces = []
        words = []
        for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(text):
            for start, end in TreebankWordTokenizer().span_tokenize(text[sent_start:sent_end]):
                words.append(text[start+sent_start:end+sent_start])
                if end+sent_start < sent_end and text[end+sent_start] == ' ':
                    spaces.append(True)
                else:
                    spaces.append(False)
        return Doc(self.vocab, words=words, spaces=spaces)

class Reader(object):
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)
        #spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm", exclude=['parser', 'tok2vec'])
        self.nlp.tokenizer = NLTKCustomTokenizer(self.nlp.vocab)
        self.feature_dim = 6
        self.aligner = Aligner(self.embedder.dimensions, self.feature_dim)
        self.train = False

    def set_spacy_embedder(self) -> None:
        start = time.time()
        new_vectors = Vectors(shape=(len(self.embedder.vocab), self.embedder.dimensions),
                              keys=self.embedder.vocab,
                              data=np.array(self.embedder.embeddings))
        self.nlp.vocab.vectors = new_vectors
        end = time.time()
        print("Setting spacy took", end - start, "seconds")

    def test_reader(self, documents: list[str], query: str, checkpoint_dir: str = None, max_words: int = 59):
        #tf.compat.v1.disable_eager_execution()
        self.aligner.load_checkpoint(checkpoint_dir)
        #self.aligner.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #print(tf.train.latest_checkpoint(checkpoint_dir))
        self.aligner.restore_checkpoint(checkpoint_dir)

        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        self.aligner.start_pred.summary()
        self.aligner.end_pred.summary()

        paragraphs = []
        all_words = []
        for i in range(len(documents)):
            paragraph_list = list(filter(None, documents[i].split('\n\n')))
            #words = list(self.__get_tokenized(retrieved_docs[j]))
            words = self.flatten([self.__get_tokenized(paragraph) for paragraph in paragraph_list])
            paragraphs += paragraph_list
            all_words += words
        
        query_tokens = TreebankWordTokenizer().tokenize(query)
        matrix_arr = tf.convert_to_tensor(self.__construct_vectors(paragraphs, all_words, query_tokens))
        
        zeroes = np.zeros((max_words - len(query_tokens), self.embedder.dimensions))
        query_embedding = tf.convert_to_tensor(np.concatenate((zeroes, np.array(list(map(self.embedder.embed, query_tokens)))), dtype=np.float32).reshape((max_words, self.embedder.dimensions, 1)))
        p_embeddings = np.array(list(map(self.embedder.embed, all_words)), dtype=np.float32)
        p_embeddings_shaped = tf.convert_to_tensor(p_embeddings.reshape((p_embeddings.shape[0], p_embeddings.shape[1], 1)))

        start, end, sp, ep = self.aligner.predict(query_embedding, p_embeddings_shaped, matrix_arr)
        print(' '.join(all_words[int(start):int(end)]))
        print(start, end)
        print(sp)
        print(ep)
    
    def train_reader2(self, doc_retriever: Retriever, documents: list[str], questions: list[tuple[str, int]], answers: list[tuple[int, int]], 
                     num_docs = 5, num_questions: int = 20, checkpoint_dir: str = None) -> None:
        questions_list = [TreebankWordTokenizer().tokenize(question[0]) for question in questions]
        max_words = max(len(q_list) for q_list in questions_list)
        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        self.aligner.start_pred.summary()
        self.aligner.end_pred.summary()

        self.aligner.load_checkpoint(checkpoint_dir)

        self.set_spacy_embedder()

        #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        query_embedding_list = []
        p_embeddings_shaped_list = []
        matrix_arr_list = []
        answer_spans_list = []
        
        index_list = list(range(len(questions)))
        random.shuffle(index_list)
        ind = 0
        start = time.time()
        for i in index_list[:num_questions]:
            print(questions[i])
            correct_doc = documents[questions[i][1]]
            correct_title = correct_doc[:correct_doc.find('\n\n\n')]
            retrieved_docs = [documents[j] for j in doc_retriever.retrieve_docs(questions[i][0], num_docs)]
            correct_ind = -1
            titles = [d[:d.find('\n\n\n')] for d in retrieved_docs]
            for j in range(num_docs):
                if titles[j] == correct_title:
                    correct_ind = j
                    break;
            if correct_ind == -1:
                retrieved_docs.pop()
                retrieved_docs.append(correct_doc)
                correct_ind = num_docs - 1
            
            #print("correct ind:", correct_ind)
            paragraphs = []
            all_words = []
            correct_spans = []
            doc_offset = 0
            token_lengths = 0
            correct_diff = 0
            for j in range(num_docs):
                paragraph_list = list(filter(None, retrieved_docs[j].split('\n\n')))
                #words = list(self.__get_tokenized(retrieved_docs[j]))
                words = self.flatten([self.__get_tokenized(paragraph) for paragraph in paragraph_list])
                if j == correct_ind:
                    doc_offset = token_lengths
                    for k in range(len(paragraph_list)):
                        correct_spans += self.__get_tokenized_spans(paragraph_list[k], inc=correct_diff)
                        correct_diff += len(paragraph_list[k]) + 2
                paragraphs += paragraph_list
                all_words += words
                token_lengths += len(words)

            print("num paragraphs:", len(paragraphs))
            print("num words:", len(all_words))
            print("answers:")
            answer_spans = []
            #words = list(self.__get_tokenized(retrieved_docs[j]))
            #token_spans = self.__get_tokenized_spans(correct_doc)
            for answer in answers[i]:
                #print("okay looking")
                print(answer)
                answer_span = self.__get_answer_span(answer, correct_spans, correct_doc)
                #print("doc offset:", doc_offset)
                print((doc_offset+answer_span[0], doc_offset+answer_span[1]+1))
                #print('\t', [all_words[doc_offset+j] for j in range(answer_span[0], answer_span[1]+1)])
                #print("test:", all_words[doc_offset-1], all_words[doc_offset], all_words[doc_offset+1])
                print('\t', all_words[doc_offset+answer_span[0]:doc_offset+answer_span[1]+1])
                print('\t', correct_doc[answer[0]: answer[1]])
                answer_spans.append((answer_span[0]+doc_offset, answer_span[1]+doc_offset+1))
            #answer_spans = tf.convert_to_tensor(answer_spans, dtype=tf.float32)
            print("Real indices:", answer_spans)
            #return;
            constructed_vectors = self.__construct_vectors(paragraphs, all_words, questions_list[i], embed=True)
            matrix_arr = constructed_vectors[:, -6:]
            p_embeddings = constructed_vectors[:, :-6]
            # print("construct shape:", constructed_vectors.shape)
            # print("matrix shape:", matrix_arr.shape)
            # print("p_embed shape:", p_embeddings.shape)
            #return;
            print("lagging part")
            zeroes = np.zeros((max_words - len(questions_list[i]), self.embedder.dimensions))
            #print("lag 1")
            query_embedding = np.concatenate((zeroes, np.array(list(map(self.embedder.embed, questions_list[i])))), dtype=np.float32).reshape((max_words, self.embedder.dimensions, 1))
            #print("lag 2")
            #p_embeddings = np.array(list(map(self.embedder.embed, all_words)), dtype=np.float32)
            #print("lag 3")
            p_embeddings_shaped = p_embeddings.reshape((p_embeddings.shape[0], p_embeddings.shape[1], 1))
            #print("lag 4")
            #return;

            query_embedding_list.append(query_embedding)
            p_embeddings_shaped_list.append(p_embeddings_shaped)
            matrix_arr_list.append(matrix_arr)
            answer_spans_list.append(answer_spans)
            print("correct ind:", correct_ind)

            if (ind + 1) % int(num_questions / 10) == 0:
                print("ML Stuff")
                for _ in range(10): print("V")
                #print("Time taken:", end - start, "seconds")
                #time.sleep(5)
                for mind in range(len(query_embedding_list)):
                    loss = self.aligner.train_step(tf.convert_to_tensor(query_embedding_list[mind]), tf.convert_to_tensor(p_embeddings_shaped_list[mind]), tf.convert_to_tensor(matrix_arr_list[mind]), tf.convert_to_tensor(answer_spans_list[mind], dtype=tf.float32))
                    print("--")
                    print("loss:", loss)
                    print("--")
                    
                print("saving.....")
                self.aligner.save_checkpoint()
                print("-----")
                query_embedding_list = []
                p_embeddings_shaped_list = []
                matrix_arr_list = []
                answer_spans_list = []
            
            ind += 1
        
        end = time.time()
        
        # print("ML Stuff")
        # for i in range(10): print("V")
        print("Time taken total:", end - start, "seconds")

    def train_reader(self, doc_retriever: Retriever, documents: list[str], questions: list[tuple[str, int]], answers: list[tuple[int, int]], 
                     num_docs = 5, num_questions: int = 1, checkpoint_dir: str = None) -> None:
        questions_list = [TreebankWordTokenizer().tokenize(question[0]) for question in questions]
        max_words = max(len(q_list) for q_list in questions_list)
        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        self.aligner.start_pred.summary()
        self.aligner.end_pred.summary()

        self.aligner.load_checkpoint(checkpoint_dir)

        self.aligner.restore_checkpoint(checkpoint_dir)

        #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        for qind in range(len(questions)):
            if questions[qind][0] == "When did the Narvaez expedition explore Florida?":
                break;
        
        # index_list = list(range(len(questions)))
        # random.shuffle(index_list)
        ind = 0
        for i in [qind]:
            correct_doc = documents[questions[i][1]]
            correct_title = correct_doc[:correct_doc.find('\n\n\n')]
            retrieved_docs = [documents[j] for j in doc_retriever.retrieve_docs(questions[i][0], num_docs)]
            correct_ind = -1
            titles = [d[:d.find('\n\n\n')] for d in retrieved_docs]
            for j in range(num_docs):
                if titles[j] == correct_title:
                    correct_ind = j
                    break;
            if correct_ind == -1:
                retrieved_docs.pop()
                retrieved_docs.append(correct_doc)
                correct_ind = num_docs - 1
            
            #print("correct ind:", correct_ind)
            paragraphs = []
            all_words = []
            correct_spans = []
            doc_offset = 0
            token_lengths = 0
            correct_diff = 0
            for j in range(num_docs):
                paragraph_list = list(filter(None, retrieved_docs[j].split('\n\n')))
                #words = list(self.__get_tokenized(retrieved_docs[j]))
                words = self.flatten([self.__get_tokenized(paragraph) for paragraph in paragraph_list])
                if j == correct_ind:
                    doc_offset = token_lengths
                    for k in range(len(paragraph_list)):
                        correct_spans += self.__get_tokenized_spans(paragraph_list[k], inc=correct_diff)
                        correct_diff += len(paragraph_list[k]) + 2
                paragraphs += paragraph_list
                all_words += words
                token_lengths += len(words)

            print("num paragraphs:", len(paragraphs))
            print("num words:", len(all_words))
            print(questions[i])
            print("answers:")
            answer_spans = []
            #words = list(self.__get_tokenized(retrieved_docs[j]))
            #token_spans = self.__get_tokenized_spans(correct_doc)
            for answer in answers[i]:
                #print("okay looking")
                print(answer)
                answer_span = self.__get_answer_span(answer, correct_spans, correct_doc)
                #print("doc offset:", doc_offset)
                print((doc_offset+answer_span[0], doc_offset+answer_span[1]+1))
                #print('\t', [all_words[doc_offset+j] for j in range(answer_span[0], answer_span[1]+1)])
                #print("test:", all_words[doc_offset-1], all_words[doc_offset], all_words[doc_offset+1])
                print('\t', all_words[doc_offset+answer_span[0]:doc_offset+answer_span[1]+1])
                print('\t', correct_doc[answer[0]: answer[1]])
                answer_spans.append((answer_span[0]+doc_offset, answer_span[1]+doc_offset+1))
            answer_spans = tf.convert_to_tensor(answer_spans, dtype=tf.float32)
            print("Real indices:", answer_spans)
            #return;
            matrix_arr = tf.convert_to_tensor(self.__construct_vectors(paragraphs, all_words, questions_list[i]))
            
            zeroes = np.zeros((max_words - len(questions_list[i]), self.embedder.dimensions))
            query_embedding = tf.convert_to_tensor(np.concatenate((zeroes, np.array(list(map(self.embedder.embed, questions_list[i])))), dtype=np.float32).reshape((max_words, self.embedder.dimensions, 1)))
            p_embeddings = np.array(list(map(self.embedder.embed, all_words)), dtype=np.float32)
            p_embeddings_shaped = tf.convert_to_tensor(p_embeddings.reshape((p_embeddings.shape[0], p_embeddings.shape[1], 1)))

            loss, start_ind, end_ind = self.aligner.train_step(query_embedding, p_embeddings_shaped, matrix_arr, answer_spans)

            print("--")
            print("loss:", loss)
            print("predicted:", (start_ind, end_ind))
            print("--")

            # if (ind + 1) % int(num_questions / 5) == 0:
            #     print("saving.....")
            #     self.aligner.save_checkpoint()
            #print("saving.....")

            #self.aligner.save_checkpoint()

            print("-----")
            ind += 1
            #return;

    def __get_answer_span(self, answer: tuple[int, int], spans: Iterator[tuple[int, int]], correct_doc: str) -> tuple[int, int]:
        start = -1
        end = -1
        ind = 0
        #print(answer)
        for span in spans:
            if start == -1 and span[0] >= answer[0]:
                #print("start stuff:", span, correct_doc[span[0]:span[1]])
                if span[0] == answer[0]: 
                    start = ind
                else:
                    print("not exact start")
                    start = ind - 1
            if span[1] >= answer[1]:
                #print("end stuff:", span, correct_doc[span[0]:span[1]])
                if span[1] != answer[1]: 
                    print("not exact end")
                end = ind
                break;
            ind += 1
        return start, end
    
    def __get_tokenized_spans(self, corpus: str, inc: int = 0):
        return self.flatten(((start+sent_start+inc, end+sent_start+inc) for start, end in TreebankWordTokenizer().span_tokenize(corpus[sent_start:sent_end])) for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(corpus))

    def __construct_vectors(self, paragraphs: list[str], all_words: list[str], query_list: list[str], embed: bool = False) -> np.ndarray:
        matrix_list = []
        start = time.time()

        counter = Counter(all_words)
        matrix_list = self.preprocess_parallel(paragraphs, len(paragraphs), set(query_list), counter, embed)
        matrix_arr = np.vstack(matrix_list)
        matrix_arr = matrix_arr.reshape(matrix_arr.shape[0], matrix_arr.shape[1], 1)
        matrix_arr[:, -3:] = matrix_arr[:, -3:]/np.linalg.norm(matrix_arr[:, -3:], axis=1)[:, None]
        
        end = time.time()
        print("took", end - start, "seconds")
        return matrix_arr.astype(np.float32)

    def __featurize(self, doc: Doc, query_set: set[str], counter: Counter, embed: bool) -> list[np.ndarray]:
        matrix = []
        for token in doc:
            match_vec = self.__exact_match(token, query_set)
            token_vec = self.__token_feature(token, counter[token.text])
            if embed:
                matrix.append(np.concatenate((token.vector, match_vec, token_vec)))
            else:
                matrix.append(np.concatenate((match_vec, token_vec)))
        return matrix
    
    def chunker(self, iterable, total_length, chunksize) -> Generator:
        return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))
    
    def flatten(self, list_of_lists: list[list]) -> list:
        return [item for sublist in list_of_lists for item in sublist]
        #return list(chain.from_iterable(list_of_lists))
    
    def process_chunk(self, texts, query_set: set[str], counter: Counter, embed: bool) -> list[list[np.ndarray]]:
        matrix_list = []
        for doc in self.nlp.pipe(texts, batch_size=20):
            matrix_list.append(self.__featurize(doc, query_set, counter, embed))
        return matrix_list
    
    def preprocess_parallel(self, texts, num_texts: int, query_set: set[str], counter: Counter, embed: bool, chunksize: int = 80, n_jobs: int = 7) -> list[np.ndarray]:
        executor = Parallel(n_jobs=n_jobs, backend='threading', prefer='processes', max_nbytes=None)
        do = delayed(self.process_chunk)
        tasks = (do(chunk, query_set, counter, embed) for chunk in self.chunker(texts, num_texts, chunksize=chunksize))
        result = executor(tasks)
        return self.flatten(result)

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        tf.compat.v1.disable_eager_execution()
        joined_docs = ' '.join(documents)
        all_words = self.__get_tokenized(joined_docs)
        print("nltk done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)
    
    def __get_tokenized(self, corpus: str) -> Iterator:
        return chain.from_iterable(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(corpus))

    def __exact_match(self, word: Token, question: set[str]) -> np.ndarray:
        og = int(word.text in question)
        low = int(word.text.lower() in question)
        lem = int(word.lemma_ in question)
        return np.array([og, low, lem])
    
    def __token_feature(self, word: Token, freq: int) -> np.ndarray:
        pos = self.nlp.meta['labels']['tagger'].index(word.tag_)
        if word.ent_type_ != "":
            ner = self.nlp.meta['labels']['ner'].index(word.ent_type_) + 1
        else:
            ner = 0
        return np.array([pos, ner, freq])
