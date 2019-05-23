# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog


def extracao(image, thresh):

	vetCarac = np.array([])

	#__________________ HU __________________#

	moments = cv2.moments(thresh)
	huMoments = cv2.HuMoments(moments)

	vetCarac = np.concatenate((vetCarac,huMoments), axis = None)


	#__________________ LBP __________________#

	radius = 5
	n_points = 3 * radius

	lbp = local_binary_pattern(thresh, n_points, radius, method='uniform')
	n_bins = int(lbp.max() + 1)
	hist, _ = np.histogram(lbp, bins = n_bins, range = (0, n_bins))

	vetCarac = np.concatenate((vetCarac,hist), axis = None)


	#__________________ HOG __________________#

	'''
	carc, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
	vetCarac = np.concatenate((vetCarac,carc), axis = None)
	'''



	#__________________ GABOR __________________#


	g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

	print(g_kernel)

	vetCarac = np.concatenate((vetCarac,gabor), axis = None)



	#__________________ CONVEX HULL __________________#

	if cv2.__version__.startswith('3.'):
		img2, contornos, hierarquia = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#Cada contorno é armazenado como um vetor de pontos.
	else:
		contornos, hierarquia = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for i in range(len(contornos)):
	    hull = cv2.convexHull(contornos[i])


	vetCarac = np.concatenate((vetCarac,hull), axis = None)
