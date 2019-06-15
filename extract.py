# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from skimage.feature import local_binary_pattern


def extracao(thresh):

	vetCarac = np.array([])
	vetCarac = np.concatenate((vetCarac,huMoments(thresh)), axis = None)
	# vetCarac = np.concatenate((vetCarac,LBP(thresh)), axis = None)
		
	return(vetCarac)



def huMoments(thresh):

	moments = cv2.moments(thresh)
	huMoments = cv2.HuMoments(moments)

	for i in range(0,7):
  		huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

	return(huMoments)



def LBP(thresh):

	radius = 5
	n_points = 3 * radius

	lbp = local_binary_pattern(thresh, n_points, radius, method='uniform')
	n_bins = int(lbp.max() + 1)
	hist, _ = np.histogram(lbp, bins = n_bins, range = (0, n_bins))

	return(hist)
