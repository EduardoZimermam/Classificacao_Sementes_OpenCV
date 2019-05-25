import matplotlib as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def classifica(vetCarac):	

	X1,y1 = make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state= 100)

	KMEANS = KMeans(n_clusters=7).fit_predict(X1)

	dbscan = DBSCAN(eps=0.5, min_samples=7, metric_params=None).fit_predict(X1)

	# plt.subplot(2,1,1)
	# plt.scatter(X1[:, 0], X1[:, 1], c= KMEANS, edgecolor='k', alpha=0.9)
	# plt.title("K-MEANS vs DBSCAN")
	# plt.subplot(2,1,2)
	# plt.scatter(X1[:, 0], X1[:, 1], c=dbscan, edgecolor='k', alpha=0.9)


	# plt.show()

	return(X1, KMEANS, dbscan)