# -*- coding: utf-8 -*-

from segm import segmentacao
from extract import extracao
from classificacao import classifica
import cv2

if __name__ == '__main__':


	image = cv2.imread('images/Emex_australis_06.jpg')
	
	segmentacao(image)

	# separacao()

	# extracao(image, thresh)

	# classifica(vetCarac)