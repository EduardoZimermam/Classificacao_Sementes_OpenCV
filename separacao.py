# -*- coding: utf-8 -*-

import numpy as np
import cv2

def subimages(thresh, nome):
	

	img = thresh.copy()


	if cv2.__version__.startswith('3.'):
		_, contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	vet_tam = []

	for i,_ in enumerate(contornos):
		_, _, width, height = cv2.boundingRect(contornos[i])
		vet_tam.append(height)
		vet_tam.append(width)
			
	limiar = 0.4*np.mean(vet_tam) #media de largura e altura
	
	for i,_ in enumerate(contornos):
		x, y, width, height = cv2.boundingRect(contornos[i])
		if width and height > limiar:
			roi = thresh[y:y+height, x:x+width]
			cv2.imwrite('subpictures/'+ nome + str(i) +'.png', roi)
