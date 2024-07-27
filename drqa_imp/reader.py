from drqa_imp.aligner import Aligner
import csv
import numpy as np
from npy_append_array import NpyAppendArray
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer
from nltk import TreebankWordTokenizer, PunktSentenceTokenizer
import spacy
from spacy.vocab import Vocab
from spacy.tokens import Token, Doc
from spacy.vectors import Vectors
from collections import Counter
from itertools import chain, groupby
import random
from collections.abc import Iterator, Generator
import warnings
import time
from joblib import Parallel, delayed
from os import path
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
        self.unknown_vec = self.__get_unknown_vector()
        self.dimensions = self.unknown_vec.shape[0]
        print(self.dimensions)
    
    def embed(self, word: str) -> np.ndarray:
        try:
            ind = self.vocab.index(word)
            return self.embeddings[ind]
        except ValueError:
            #print(word)
            return self.unknown_vec

    def fine_tune(self, words: list[str], documents: list[str], max_iterations: int = 1000, vocab_save: str|None = None, embed_save: str|None = None) -> None:
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

    def __get_unknown_vector(self) -> np.ndarray:
        return np.array("0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811"
                        .split(" "), dtype=np.float32)

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
    def __init__(self, vocab_path: str = None, embed_path: str = None, glove_path: str = None, max_q: int = -1, max_p: int = -1, batch_size: int = 32) -> None:
        self.embedder = Embedder(vocab_path, embed_path, glove_path)
        self.nlp = spacy.load("en_core_web_sm", exclude=['parser'])
        self.nlp.tokenizer = NLTKCustomTokenizer(self.nlp.vocab)
        self.feature_dim = 6
        self.max_q = max_q
        self.max_p = max_p
        self.aligner = Aligner(self.embedder.dimensions, self.max_q, self.max_p)
        self.batch_size = batch_size

    def set_spacy_embedder(self) -> None:
        start = time.time()
        new_vectors = Vectors(shape=(len(self.embedder.vocab), self.embedder.dimensions),
                              keys=self.embedder.vocab,
                              data=np.array(self.embedder.embeddings))
        self.nlp.vocab.vectors = new_vectors
        end = time.time()
        print("Setting spacy took", end - start, "seconds")

    def test_reader(self, documents: list[str], query: str, max_q: int, max_p: int, checkpoint_dir: str = None):
        self.aligner.load_checkpoint(checkpoint_dir)
        self.aligner.restore_checkpoint(checkpoint_dir)
        self.set_spacy_embedder()

        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        self.aligner.start_pred.summary()
        self.aligner.end_pred.summary()

        query_tokens = TreebankWordTokenizer().tokenize(query)
        query_embedding = np.array(list(map(self.embedder.embed, query_tokens)), dtype=np.float32)
        query_embedding = self.__pad_array(query_embedding, max_q)

        query_embedding_list = []
        p_embeddings_list = []
        matrix_arr_list = []
        all_word_spans = []
        pad_vals = []
        for i in range(len(documents)):
            paragraph_inds = list(self.__split_inds(documents[i], '\n'))
            paragraph_list = [documents[i][inds[0]:inds[1]] for inds in paragraph_inds]
            words = self.__flatten([self.__get_tokenized(paragraph) for paragraph in paragraph_list])
            word_spans = self.__flatten([self.__get_tokenized_spans(paragraph_list[k], inc=paragraph_inds[k][0]) for k in range(len(paragraph_list))])
            all_word_spans.append(word_spans)
        
            constructed_vectors = self.__construct_vectors(paragraph_list, words, query_tokens, embed=True)
            matrix_arr = constructed_vectors[:, -6:]
            p_embeddings = constructed_vectors[:, :-6]

            matrix_arr = self.__pad_array(matrix_arr, max_p)
            pad_vals.append(max_p - p_embeddings.shape[0])
            p_embeddings = self.__pad_array(p_embeddings, max_p)
            
            query_embedding_list.append(query_embedding)
            p_embeddings_list.append(p_embeddings)
            matrix_arr_list.append(matrix_arr)

        start, end, start_argmax, end_argmax = self.aligner.predict(np.array(query_embedding_list), np.array(p_embeddings_list), 
                                                  np.array(matrix_arr_list))
        ind = int(start[0])
        if ind != int(end[0]):
            raise ValueError("something is muy muy fishy")
        
        cspan_start = all_word_spans[ind][int(start[1]) - pad_vals[ind]]
        cspan_end = all_word_spans[ind][int(end[1]) - pad_vals[ind]]
        print("answer:", documents[ind][cspan_start[0]:cspan_end[1]])
        print("doc title:", documents[ind][:documents[ind].find('\n\n\n')])
        print('------')
        print("other starts:")
        for i in range(len(documents)):
            print(documents[i][all_word_spans[i][int(start_argmax[i]) - pad_vals[i]][0]:all_word_spans[i][int(end_argmax[i]) - pad_vals[i]][1]])
            print('---')
    
    def prepare_training_data(self, documents: list[str], questions: list[tuple[str, int]], answers: list[tuple[int, int]], save_dir: str,
                              num_questions: int = 10000) -> None:
        self.set_spacy_embedder()
        max_q = 0
        max_p = 0

        index_list = list(range(len(questions)))
        random.shuffle(index_list)
        
        ind = 0
        query_list = [TreebankWordTokenizer().tokenize(questions[i][0]) for i in index_list[:num_questions]]
        max_q = max((len(query) for query in query_list))
        paragraph_inds_list = [list(self.__split_inds(documents[questions[i][1]], '\n')) for i in index_list[:num_questions]]
        paragraph_lists = [[documents[questions[index_list[i]][1]][inds[0]:inds[1]] for inds in paragraph_inds_list[i]] for i in range(num_questions)]
        words_list = [self.__flatten(self.__get_tokenized(paragraph) for paragraph in paragraph_list) for paragraph_list in paragraph_lists]
        max_p = max((len(words) for words in words_list))
        print("Max q:", max_q, "Max p:", max_p)

        q_path = path.join(save_dir, f"queries_{num_questions}.npy")
        p_path = path.join(save_dir, f"paragraphs_{num_questions}.npy")
        f_path = path.join(save_dir, f"feature_matrices_{num_questions}.npy")
        s_path = path.join(save_dir, f"start_spans_{num_questions}.npy")
        e_path = path.join(save_dir, f"end_spans_{num_questions}.npy")
        sizes_path = path.join(save_dir, "sizes.csv")
        
        start = time.time()

        with open(sizes_path, 'w') as f:
            f.write("max_q,max_p,questions\n")
            f.write(f"{max_q},{max_p},{num_questions}")
        
        with NpyAppendArray(q_path) as q_file, NpyAppendArray(p_path) as p_file, NpyAppendArray(f_path) as f_file, NpyAppendArray(s_path) as s_file, NpyAppendArray(e_path) as e_file:
            for i in index_list[:num_questions]:
                max_q = max(max_q, len(query_list[ind]))
                print(questions[i])

                correct_doc = documents[questions[i][1]]
                paragraph_list = paragraph_lists[ind]
                paragraph_inds = paragraph_inds_list[ind]
                all_words = words_list[ind]
                max_p = max(max_p, len(all_words))
                correct_spans = self.__flatten([self.__get_tokenized_spans(paragraph_list[k], inc=paragraph_inds[k][0]) for k in range(len(paragraph_list))])

                print("num paragraphs:", len(paragraph_list))
                print("num words:", len(all_words))
                print("answers:")
                start_spans = np.zeros(len(all_words))
                end_spans = np.zeros(len(all_words))
                for answer in answers[i]:
                    print(answer)
                    answer_span = self.__get_answer_span(answer, correct_spans)
                    print((answer_span[0], answer_span[1]+1))
                    print('\t', all_words[answer_span[0]:answer_span[1]+1])
                    print('\t', correct_doc[answer[0]: answer[1]])
                    start_spans[answer_span[0]] = 1
                    end_spans[answer_span[1]] = 1
                    diff = answer_span[1]+1 - answer_span[0]
                    for j in range(len(all_words)):
                        if all_words[j] == all_words[answer_span[0]] and j != answer_span[0] and all_words[j:j+diff] == all_words[answer_span[0]:answer_span[1]+1]:
                            print("match found:", all_words[j:j+diff])
                            start_spans[j] = 1
                            end_spans[j+diff-1] = 1
                    print("Real indices:", (answer_span[0], answer_span[1]))
                
                constructed_vectors = self.__construct_vectors(paragraph_list, all_words, query_list[ind], embed=True)
                matrix_arr = constructed_vectors[:, -6:]
                p_embeddings = constructed_vectors[:, :-6]
                query_embedding = np.array(list(map(self.embedder.embed, query_list[ind])))
                
                q_file.append(self.__pad_array(query_embedding, max_q))
                p_file.append(self.__pad_array(p_embeddings, max_p))
                f_file.append(self.__pad_array(matrix_arr, max_p))
                s_file.append(self.__pad_array(start_spans, max_p))
                e_file.append(self.__pad_array(end_spans, max_p))
                ind += 1
        
        end = time.time()
        print("Done loading. Time taken total:", end - start, "seconds")
        print("Max q:", max_q, "Max p:", max_p)
    
    def __pad_array(self, array: np.ndarray, max_val: int) -> np.ndarray:
        pad_len = max_val - array.shape[0]
        pad_arr = [(pad_len, 0)] + [(0, 0) for _ in range(array.ndim - 1)]
        return np.pad(array, pad_arr, 'constant')
    
    def train_reader(self, data_dir: str, count: int, checkpoint_dir: str) -> None:
        self.aligner.q_encoder.summary()
        self.aligner.q_aligner.summary()
        self.aligner.start_pred.summary()
        self.aligner.end_pred.summary()

        self.aligner.load_checkpoint(checkpoint_dir)

        queries = np.load(path.join(data_dir, f"queries_{count}.npy"), mmap_mode='r').reshape(count, self.max_q, self.embedder.dimensions)
        paragraphs = np.load(path.join(data_dir, f"paragraphs_{count}.npy"), mmap_mode='r').reshape(count, self.max_p, self.embedder.dimensions)
        feature_matrices = np.load(path.join(data_dir, f"feature_matrices_{count}.npy"), mmap_mode='r').reshape(count, self.max_p, 6)
        start_spans = np.load(path.join(data_dir, f"start_spans_{count}.npy"), mmap_mode='r').reshape(count, self.max_p)
        end_spans = np.load(path.join(data_dir, f"end_spans_{count}.npy"), mmap_mode='r').reshape(count, self.max_p)

        for i in range(0, count, self.batch_size):
            loss = self.aligner.train_step(tf.convert_to_tensor(queries[i:i+self.batch_size]), tf.convert_to_tensor(paragraphs[i:i+self.batch_size]), 
                                            tf.convert_to_tensor(feature_matrices[i:i+self.batch_size]), tf.convert_to_tensor(start_spans[i:i+self.batch_size], dtype=tf.float32),
                                            tf.convert_to_tensor(end_spans[i:i+self.batch_size], dtype=tf.float32))
            print("--")
            print("loss:", loss)
            print("--")
        print("saving.....")
        self.aligner.save_checkpoint()
        print("-----")

    def __split_inds(self, text: str, separator: str = ' ') -> Generator[tuple[int, int]]:
        start = 0
        for key, group in groupby(text, lambda x: x[:len(separator)]==separator):
            end = start + sum(1 for _ in group)
            if not key:
                yield start, end
            start = end

    def __get_answer_span(self, answer: tuple[int, int], spans: Iterator[tuple[int, int]]) -> tuple[int, int]:
        start = -1
        end = -1
        ind = 0
        for span in spans:
            if start == -1 and span[0] >= answer[0]:
                if span[0] == answer[0]: 
                    start = ind
                else:
                    print("not exact start")
                    start = ind - 1
            if span[1] >= answer[1]:
                if span[1] != answer[1]: 
                    print("not exact end")
                end = ind
                break;
            ind += 1
        return start, end
    
    def __get_tokenized_spans(self, corpus: str, inc: int = 0):
        return self.__flatten(((start+sent_start+inc, end+sent_start+inc) for start, end in TreebankWordTokenizer().span_tokenize(corpus[sent_start:sent_end])) for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(corpus))

    def __construct_vectors(self, paragraphs: list[str], all_words: list[str], query_list: list[str], embed: bool = False) -> np.ndarray:
        matrix_list = []
        start = time.time()

        counter = Counter(all_words)
        matrix_list = self.__preprocess_parallel(paragraphs, len(paragraphs), set(query_list), counter, embed)

        matrix_arr = np.vstack(matrix_list)
        matrix_arr[:, -3:] = matrix_arr[:, -3:]/np.linalg.norm(matrix_arr[:, -3:], axis=0)
        
        end = time.time()
        print("took", end - start, "seconds")
        return matrix_arr.astype(np.float32)
    
    def __preprocess_parallel(self, texts, num_texts: int, query_set: set[str], counter: Counter, embed: bool, chunksize: int = 80, n_jobs: int = 7) -> list[np.ndarray]:
        executor = Parallel(n_jobs=n_jobs, backend='threading', prefer='processes', max_nbytes=None)
        do = delayed(self.__process_chunk)
        tasks = (do(chunk, query_set, counter, embed) for chunk in self.__chunker(texts, num_texts, chunksize=chunksize))
        result = executor(tasks)
        return self.__flatten(result)

    def __chunker(self, iterable, total_length, chunksize) -> Generator:
        return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

    def __process_chunk(self, texts, query_set: set[str], counter: Counter, embed: bool) -> list[list[np.ndarray]]:
        matrix_list = []
        for doc in self.nlp.pipe(texts, batch_size=20):
            matrix_list.append(self.__featurize(doc, query_set, counter, embed))
        return matrix_list

    def __featurize(self, doc: Doc, query_set: set[str], counter: Counter, embed: bool) -> list[np.ndarray]:
        matrix = []
        for token in doc:
            match_vec = self.__exact_match(token, query_set)
            token_vec = self.__token_feature(token, counter[token.text])
            # if token.is_oov:
            #     print(token.text)
            embed_vec = token.vector if not token.is_oov else self.embedder.unknown_vec
            if embed:
                matrix.append(np.concatenate((embed_vec, match_vec, token_vec)))
            else:
                matrix.append(np.concatenate((match_vec, token_vec)))
        return matrix

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
        #print(word.text, word.pos_, pos, ner, freq)
        return np.array([pos, ner, freq])

    def fine_tune_embedder(self, documents: list[str], common_count: int = 1000, vocab_save: str = None, embed_save: str = None) -> None:
        tf.compat.v1.disable_eager_execution()
        joined_docs = ' '.join(documents)
        all_words = self.__get_tokenized(joined_docs)
        print("nltk done tokenizing")
        counter = Counter(all_words)
        common_words, counts = zip(*counter.most_common(common_count))
        self.embedder.fine_tune(common_words, documents, max_iterations=2000, vocab_save=vocab_save, embed_save=embed_save)
    
    def __get_tokenized(self, corpus: str) -> Iterator:
        return self.__flatten(TreebankWordTokenizer().tokenize(sentence) for sentence in PunktSentenceTokenizer().tokenize(corpus))

    def __flatten(self, list_of_lists: list[list]) -> list:
        return [item for sublist in list_of_lists for item in sublist]
