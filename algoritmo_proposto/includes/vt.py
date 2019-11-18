'''
Vocabulary Tree
'''


import vv
from clust import KMeans
import numpy as np


def read_features(file_name, path='./', inverted_files=vv.InvertedFiles()):
    list_features = []
    file = open(f'{path}{file_name}')
    name = file.readline().replace('\n', '')
    for line in file:
        feature = Feature(np.float32(line.split()))
        feature.ID = str(feature)
        inverted_files.word_file(feature.ID, file_name)
        list_features.append(feature)
    file.close()
    # try:
    #     list_features_aux = KMeans(n_clusters=1000).predict(list_features)[0]
    #     list_features = list_features_aux
    # except expression as identifier:
    #     pass
    return list_features, inverted_files


class Feature(list):

    __slots__ = 'ID'


class Node(object):#TODO 46

    __slots__ = 'list_features', 'level', 'childrens', 'inverted_files', 'W', 'M', 'TF_IDF', 'centers'

    def __init__(self, list_features, level, inverted_files, K=3, L=3, n_docs=1):
        self.list_features = list_features
        self.childrens = []
        self.level = level
        self.W, self.M = self._get_w_m(n_docs, inverted_files)
        self.TF_IDF = self._get_TF_IDF()
        if (L == 0) or (len(self.list_features) <= K):
            self.inverted_files = inverted_files
            return
        self.centers, clusters = KMeans(K).predict(self.list_features)
        self.childrens = [ Node(c, level+1, inverted_files, K=K, L=L-1, n_docs=n_docs) for c in clusters ]

    def __del__(self):
        del self.list_features
        del self.childrens
        del self.level

    def _get_w_m(self, n_docs, inverted_files):
        docs = {}
        for feature in self.list_features:
            list_docs = inverted_files.INDEX[feature.ID]
            for doc in list_docs:
                if not doc in docs:
                    docs[doc] = 1
                else:
                    docs[doc] += 1
        return np.log(n_docs / float(len(docs.keys()))), docs

    def _get_TF_IDF(self):
        TF_IDF = {}
        for doc in self.M.keys():
            TF_IDF[doc] = self.W * self.M[doc]
        return TF_IDF

    def get_score(self, doc):
        return self.TF_IDF[doc] if doc in self.TF_IDF else 0


class Tree(object):

    __slots__ = 'root'

    def __init__(self):
        self.root = None

    def __del__(self):
        del self.root

    def fit(self, list_features, inverted_files, K=3, L=3, n_docs=1):
        self.root = Node(list_features, 0, inverted_files, K=K, L=L, n_docs=n_docs)

    def get_histogram(self, doc):
        hist = []
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            for children in current.childrens:
                queue.append(children)
            hist.append(current.get_score(doc))
        return hist


def hist_compare(hist1, hist2):
    vet_result = []
    for i in range(len(hist1)):
        vet_result.append(hist1[i]/float(np.linalg.norm(hist1)) - hist2[i]/float(np.linalg.norm(hist2)))
    return np.linalg.norm(vet_result)


if __name__ == "__main__":
    l = np.random.rand(10, 4)
    tree = Tree()
    tree.fit(l.tolist())
