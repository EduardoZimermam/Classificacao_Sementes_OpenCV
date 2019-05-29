# -*- coding: utf-8 -*-

import numpy as np
import cv2

def subimages(image,thresh):
	

	img = thresh.copy()


	if cv2.__version__.startswith('3.'):
		_, contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		contornos,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	

	for i,_ in enumerate(contornos):
		x, y, width, height = cv2.boundingRect(contornos[i])
		#vetor 
		#média

		roi = thresh[y:y+height, x:x+width]
		cv2.imwrite('subpictures/'+ str(i) +'.png', roi)
