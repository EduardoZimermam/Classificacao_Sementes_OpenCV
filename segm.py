# -*- coding: utf-8 -*-

import cv2
from extract import getConvexHu

def segmentacao(img):
	
	hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	red, green, blue = cv2.split(img)

	hue, saturation, value = cv2.split(hsvImg)



	blur = cv2.GaussianBlur(saturation, (17,17), 0)


	_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)


	cv2.imshow('Imagem Binarizada', thresh)
	cv2.waitKey(0)