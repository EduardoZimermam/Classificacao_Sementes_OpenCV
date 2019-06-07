# -*- coding: utf-8 -*-

import numpy as np
import cv2

def subimages(thresh, nome):
	

	img = thresh.copy()


	if cv2.__version__.startswith('3.'):
		_, contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	vet_lar = []
	vet_alt = []

	for i,_ in enumerate(contornos):
		_, _, width, height = cv2.boundingRect(contornos[i])
		vet_alt.append(height)
		vet_lar.append(width)
			
	limiar_alt = 0.4*np.mean(vet_alt) #media de largura e altura
	limiar_lar = 0.4*np.mean(vet_lar) #media de largura e altura
	
	for i,_ in enumerate(contornos):
		x, y, width, height = cv2.boundingRect(contornos[i])
		if width > limiar_lar and height > limiar_alt:
			roi = thresh[y:y+height, x:x+width]
 			cv2.imwrite('subpictures/'+ nome + str(i) +'.png', roi)
