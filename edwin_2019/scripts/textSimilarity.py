import os
import sys
from collections import defaultdict

import jieba
import gensim
from sklearn.cluster import KMeans

FILE_DIR = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
STOP_WORD_FILE_CHN = os.path.join(FILE_DIR, "stopwords.txt")
STOP_WORD_LIST_CHN = [word.strip() for word in open(STOP_WORD_FILE_CHN, "r", encoding="utf-8").readlines()]


class TextSimilarity(object):
    def __init__(self, documents, drop_freq=2):
        """
        :param documents: list of list of words. The sublist must not be empty.
        """
        self._documents = documents
        self._dictionary = None
        self._corpus = None
        self._idx = None
        self._model = None
        self._model_name = None
        self._min_cnt = drop_freq
        self._pre_process()

    def _pre_process(self):
        texts = self._documents
        if self._min_cnt > 0:
            word_freq = defaultdict(int)
            for text in self._documents:
                for word in text:
                    word_freq[word] += 1
            texts = [[word for word in text if word_freq[word] > self._min_cnt] for text in self._documents]
        self._documents = [word_list for word_list in texts]
        self._dictionary = gensim.corpora.Dictionary(self._documents)
        self._corpus = [self._dictionary.doc2bow(doc) for doc in self._documents]

    @staticmethod
    def cut_sentence(sentence, drop_stop_word=True, language="CHN"):
        stop_word_list = []
        if drop_stop_word:
            if language == "CHN":
                stop_word_list = STOP_WORD_LIST_CHN
            else:
                raise Exception("Only support CHN.")
        return tuple(word.strip() for word in jieba.cut(sentence=sentence) if word.strip() and word.strip() not in stop_word_list)

    def select_model(self, model_name, num_topics=300):
        """
        :type model_name: basestring
        :type num_topics: int
        :return: None
        """
        if self._model is not None:
            print("Only one selection for each instance.")
            return
        self._model_name = model_name.lower()
        if model_name == "tf-idf" or model_name == "lsa" or model_name == "rp":
            self._model = gensim.models.TfidfModel(self._corpus, normalize=True)
            corpus_tfidf = self._model[self._corpus]
            if model_name == model_name == "lsa":
                self._model = gensim.models.LsiModel(corpus_tfidf, id2word=self._dictionary, num_topics=num_topics)
            elif model_name == model_name == "rp":
                self._model = gensim.models.RpModel(corpus_tfidf, num_topics=num_topics)
        elif model_name == "lda":
            self._model = gensim.models.LdaModel(self._corpus, id2word=self._dictionary, num_topics=num_topics)
        elif model_name == "hdp":
            self._model = gensim.models.HdpModel(self._corpus, id2word=self._dictionary)
        else:
            raise Exception("Invalid model name")

    def find_similar(self, query_word_bag, threshold=0.8):
        """
        :param threshold: threshold for "being similar"
        :param query_word_bag: list of words without stop word
        :type query_word_bag: list
        :type threshold: float
        :return: documents and their similarities
        """
        if not query_word_bag:
            return list()
        vec_bow = self._dictionary.doc2bow(query_word_bag)
        vec_query = self._model[vec_bow]
        if self._idx is None:
            self._idx = gensim.similarities.MatrixSimilarity(self._model[self._corpus])
        sims = self._idx[vec_query]
        sims = sorted(enumerate(sims), key=lambda x: -x[1])
        return [(self._documents[i], s[1]) for i, s in enumerate(sims) if s[1] > threshold]

    def cluster(self, n_clusters=8, workers=4):
        self._model_name = "K-Means"
        self._model = gensim.models.Word2Vec(self._documents, workers=workers, min_count=self._min_cnt)
        doc_vec = [sum(self._model[key] for key in doc) for doc in self._documents]
        # Do clustering with KMeans
        clf = KMeans(n_clusters=n_clusters)
        s = clf.fit(doc_vec)
        labels = clf.labels_

        return labels

    @property
    def similarity_index(self):
        return self._idx

    @property
    def model(self):
        return self._model_name

