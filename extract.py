# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def extracao(thresh):

	
	vetCarac = np.array([])
	vetCarac = np.concatenate((vetCarac,huMoments(thresh)), axis = None)
	# vetCarac = np.concatenate((vetCarac,LBP(thresh)), axis = None)
	# vetCarac = np.concatenate((vetCarac,HOG(thresh)), axis = None)
	
	return(vetCarac)



def huMoments(thresh):

	moments = cv2.moments(thresh)
	huMoments = cv2.HuMoments(moments)

	return(huMoments)



def LBP(thresh):

	radius = 5
	n_points = 3 * radius

	lbp = local_binary_pattern(thresh, n_points, radius, method='uniform')
	n_bins = int(lbp.max() + 1)
	hist, _ = np.histogram(lbp, bins = n_bins, range = (0, n_bins))

	return(hist)



def HOG(thresh):

	carc, hog_image = hog(thresh, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L1', visualize=True, multichannel=False)
	
	count = 0
	soma = 0 

	for x in carc:
		if(x > 0):
			soma += x
			count += 1
	
	media = soma / count
	
	return(media)
