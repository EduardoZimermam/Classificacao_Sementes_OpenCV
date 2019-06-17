# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
from glob import glob
import matplotlib.pyplot as plt

from segm import segmentacao
from separacao import subimages
from extract import extracao

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


if __name__ == '__main__':

	#se nao existir a pasta de subpictures
	if not (os.path.isdir('./subpictures')):
	  		os.mkdir('./subpictures') #criar

	if len(os.listdir('./subpictures') ) == 0:

	  	labels = []

		print('Iniciando a segmentação e subdivisão das imagens...')
		  	
		for i, pasta in enumerate(glob('Base_Imagens_PI/Trabalho_PI_2019/*')):
			labels.append(pasta.replace('Base_Imagens_PI/Trabalho_PI_2019/', ''))

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
	print('\n')


	KMEANS = KMeans(n_clusters=7).fit_predict(vetCarac)

	dbscan = DBSCAN(eps=1.0, min_samples= 10).fit(vetCarac)

	labels = dbscan.labels_
	dbscan = dbscan.fit_predict(vetCarac)

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('---------- DBSCAN ----------')
	print('Número estimado de clusters: %d' % n_clusters_)
	print('Número de pontos não classificados: %d' % n_noise_)
	print("Coeficiente de Silhouette: %0.3f" % metrics.silhouette_score(vetCarac, labels))
	print('\n')


	# Plot result
	plt.subplot(2,1,1)
	plt.scatter(vetCarac[:, 2], vetCarac[:, 3], c= KMEANS, edgecolor='k', alpha=0.9)
	plt.title("K-MEANS vs DBSCAN")
	plt.subplot(2,1,2)
	plt.scatter(vetCarac[:, 2], vetCarac[:, 3], c=dbscan, edgecolor='k', alpha=0.9)


	plt.show()
