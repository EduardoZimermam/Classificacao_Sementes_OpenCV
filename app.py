# -*- coding: utf-8 -*-

import cv2
from glob import glob
import numpy as np
from segm import segmentacao
from separacao import subimages
from extract import extracao
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from sklearn import preprocessing


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

	vetCarac = preprocessing.normalize(vetCarac)

	#classifica(vetCarac)

	
	

	vetCarac = np.array(vetCarac)

	KMEANS = KMeans(n_clusters=7).fit_predict(vetCarac)

	dbscan = DBSCAN(eps=0.5, min_samples=10, metric_params=None).fit_predict(vetCarac)

	plt.subplot(2,1,1)
	plt.scatter(vetCarac[:, 0], vetCarac[:, 1], c= KMEANS, edgecolor='k', alpha=0.9)
	plt.title("K-MEANS vs DBSCAN")
	plt.subplot(2,1,2)
	plt.scatter(vetCarac[:, 0], vetCarac[:, 1], c=dbscan, edgecolor='k', alpha=0.9)


	plt.show()