# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def classifica():	

	X1,y1 = make_blobs(n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state= 100)


	scaler = StandardScaler()
	X1 = scaler.fit_transform(X1)
	print(X1)

	# KMEANS = KMeans(n_clusters=7).fit_predict(vetCarac)

	# dbscan = DBSCAN(eps=0.5, min_samples=10, metric_params=None).fit_predict(vetCarac)

	# pyplot.subplot(2,1,1)
	# pyplot.scatter(vetCarac[:, 0], vetCarac[:, 1], c= KMEANS, edgecolor='k', alpha=0.9)
	# pyplot.title("K-MEANS vs DBSCAN")
	# pyplot.subplot(2,1,2)
	# pyplot.scatter(vetCarac[:, 0], vetCarac[:, 1], c=dbscan, edgecolor='k', alpha=0.9)


	# pyplot.show()

	