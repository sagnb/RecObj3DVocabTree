import numpy as np
from sklearn import cluster


class KMeans(object):

    # __slots__ 

    def __init__(self, n_clusters = 3):
        self.n_clusters = n_clusters

    def _clusters(self, pred, des):
        pred_des = zip(pred, des)

        clusters = [[] for _ in range(self.n_clusters)]

        for (p, d) in pred_des:
            clusters[p].append(d)
        return clusters

    def predict(self, X):
        kmeans_instance = cluster.KMeans(n_clusters=self.n_clusters, random_state=0)
        pred = kmeans_instance.fit_predict(X)
        return (kmeans_instance.cluster_centers_, np.array([elem for elem in self._clusters(pred, X)]))
