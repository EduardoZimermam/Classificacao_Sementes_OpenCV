# -*- coding: utf-8 -*-

import cv2
import numpy as np
from glob import glob
from segm import segmentacao
from separacao import subimages
from extract import extracao
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from classificacao import classifica


if __name__ == '__main__':

  	print('Iniciando a segmentação e subdivisão das imagens...')

	for pasta in glob('Trabalho_PI_2019/*'):
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


	print('Iniciando a clusterização das subimagens...')
	
	# classifica()

	scaler = StandardScaler()
	vetCarac = scaler.fit_transform(vetCarac)

	from sklearn.decomposition import PCA

	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(vetCarac)
	
	
	KMEANS = KMeans(n_clusters=7).fit_predict(X_pca)

	dbscan = DBSCAN(eps=1.0, min_samples=10).fit_predict(X_pca)

	plt.subplot(2,1,1)
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c= KMEANS, edgecolor='k', alpha=0.9)
	plt.title("K-MEANS vs DBSCAN")
	plt.subplot(2,1,2)
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan, edgecolor='k', alpha=0.9)


	plt.show()