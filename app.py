# -*- coding: utf-8 -*-

import cv2
import numpy as np
from glob import glob
from segm import segmentacao
from separacao import subimages
from extract import extracao
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os.path
from sklearn.datasets import make_blobs


if __name__ == '__main__':

	#se nao existir a pasta de subpictures
	if not (os.path.isdir('./subpictures')):
	  		os.mkdir('./subpictures') #criar

	if len(os.listdir('./subpictures') ) == 0:
		print('Iniciando a segmentação e subdivisão das imagens...')
	  	labels = []

	 	
		for i, pasta in enumerate(glob('Trabalho_PI_2019/*')):

			labels.append(pasta.replace('Trabalho_PI_2019/', ''))	
			for img in glob(pasta + '/*.jpg'):
				image = cv2.imread(img)

				nome = img.replace(pasta, '')
				nome = nome.replace('.jpg', '_')
				
				thresh = segmentacao(image)
				
				subimages(thresh, nome)
			
		print('Todas as imagens foram segmentadas e divididas...')



	print('Iniciando a extração das características das subimagens...')

	vetCarac = []

  	for i, img in enumerate(glob('subpictures/*.png')):
		
		image = cv2.imread(img)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		vetCarac.append(extracao(image))


	scaler = StandardScaler()
	vetCarac = scaler.fit_transform(vetCarac)


	print('Iniciando a clusterização das subimagens...')
	
	

	

	from sklearn.decomposition import PCA
	from sklearn import metrics

	# pca = PCA(n_components=10)
	# vetCarac = pca.fit_transform(vetCarac)

####################### DB ########################################	
	db = DBSCAN(eps=0.8, min_samples=10).fit(vetCarac)

	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(vetCarac, labels))

	# #############################################################################
	# Plot result

	# pyplot.show()
	import matplotlib.pyplot as plt

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = [0, 0, 0, 1]

	    class_member_mask = (labels == k)

	    xy = vetCarac[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 1], xy[:, 3], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=14)

	    xy = vetCarac[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 1], xy[:, 3], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()




	# from mpl_toolkits.mplot3d import Axes3D

	# # initialize figure and 3d projection for the PC3 data
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# # assign x,y,z coordinates from PC1, PC2 & PC3
	# xs = vetCarac.T[0]
	# ys = vetCarac.T[1]
	# zs = vetCarac.T[2]

	# plot = ax.scatter(xs, ys, zs, alpha=0.75,
	#                   c=dbscan, cmap='viridis', depthshade=True)


	# fig.colorbar(plot, shrink=0.6, aspect=9)
	# plt.show()

