# -*- coding: utf-8 -*-

from segm import segmentacao
import cv2

if __name__ == '__main__':


	img = cv2.imread('images/Emex_australis_06.jpg')

	segmentacao(img)

	# separacao()

	# extracao()

	# classificacao()