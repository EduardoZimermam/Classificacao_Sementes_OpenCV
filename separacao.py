# -*- coding: utf-8 -*-

import numpy as np
import cv2

def subimages(image,thresh):

	if cv2.__version__.startswith('3.'):
		img2, contornos, hierarquia = cv2.findContours(thresh, 1, 2)
	else:
		contornos, hierarquia = cv2.findContours(thresh, 1, 2)

	for x in contornos:
		cnt = x
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(image,[box],0,(0,0,255),2)

	cv2.imshow('Contours', image)
	cv2.waitKey(0)