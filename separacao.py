# -*- coding: utf-8 -*-

import numpy as np
import cv2

def subimages(thresh, nome):

	img = thresh.copy()

	if cv2.__version__.startswith('3.'):
	 	_, contornos,hierarchy = cv2.findContours(img, cv2. RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	else:
	 	contornos,hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	

	vet_lar = []
	vet_alt = []


	for i,_ in enumerate(contornos):
		_, _, width, height = cv2.boundingRect(contornos[i])
	 	vet_alt.append(height)
	 	vet_lar.append(width)
			
	limiar_alt = 0.5*np.mean(vet_alt) #media de largura e altura
	limiar_lar = 0.5*np.mean(vet_lar) #media de largura e altura
	

	hull = []

	contornos = sorted(contornos, key=lambda ctr: cv2.boundingRect(ctr)[0])


	for i in range(len(contornos)):
		black = np.zeros_like(thresh)
		hull = cv2.convexHull(contornos[i], False)
		
		#desenha contornos e separa cada imagem
		cv2.drawContours(black, [hull], -1, (255, 255, 255), -1)
		r, t2 = cv2.threshold(black, 127, 255, cv2.THRESH_BINARY)
		masked = cv2.bitwise_and(thresh, thresh, mask = t2)
	

		#corta a imagem em um retangulo menor
		_, _, width, height = cv2.boundingRect(contornos[i])

		if (width > limiar_lar) and (height > limiar_alt):
			x, y, width, height = cv2.boundingRect(contornos[i])
			roi = masked[y:y+height, x:x+width]
			cv2.imwrite('subpictures/'+ nome + str(i) +'.png', roi)
		