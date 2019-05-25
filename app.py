# -*- coding: utf-8 -*-

import cv2
import numpy as np
from segm import segmentacao
from separacao import subimages
from extract import extracao
from classificacao import classifica

import matplotlib.pyplot as plt


if __name__ == '__main__':


	image = cv2.imread('images/Emex_australis_06.jpg')
	
	thresh = segmentacao(image)

	subimages(image, thresh)


	vetCarac = np.array([])
	vetCarac = extracao(thresh)
	
	#----- DANDO ERRO (VER) ----#

	X1, KMEANS, dbscan = classifica(vetCarac)

	plt.subplot(2,1,1)
	plt.scatter(X1[:, 0], X1[:, 1], c= KMEANS, edgecolor='k', alpha=0.9)
	plt.title("K-MEANS vs DBSCAN")
	plt.subplot(2,1,2)
	plt.scatter(X1[:, 0], X1[:, 1], c=dbscan, edgecolor='k', alpha=0.9)


	plt.show()