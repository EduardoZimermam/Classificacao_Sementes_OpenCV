# -*- coding: utf-8 -*-

import cv2

def segmentacao(img):
	
	grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, thresh = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU)

	cv2.imshow('Imagem Binarizada', thresh)
	cv2.waitKey(0)


