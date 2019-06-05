# -*- coding: utf-8 -*-

import cv2

def segmentacao(img):
	
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	red, green, blue = cv2.split(img)

	hue, saturation, value = cv2.split(hsvImg)

	blur = cv2.GaussianBlur(saturation, (25,25), 0)

	_, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

	thresh = thresh * imgGray


	# cv2.imshow('Imagem Binarizada', thresh)
	# cv2.waitKey(0)

	return(thresh)