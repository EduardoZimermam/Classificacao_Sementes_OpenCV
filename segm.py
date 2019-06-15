# -*- coding: utf-8 -*-

import cv2
import numpy as np

def segmentacao(img):
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	imgGray = cv2.cvtColor(hsvImg, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((11,11),np.uint8)

	blur = cv2.GaussianBlur(imgGray, (11,11), 0)

	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	_, thresh = cv2.threshold(blur, 163, 255, cv2.THRESH_BINARY_INV)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	

	thresh = thresh * gray

	return(thresh)