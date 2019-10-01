'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
from roipoly import RoiPoly
from matplotlib import pyplot as plt 
import numpy as np
import matplotlib.patches as mpatches
import barrel_detector_train

if __name__ == '__main__':
	
	folder = "testset"	

	# Test Model

	W = np.load("weights1.npy")
	b = np.load("bias1.npy")
	my_detector = BarrelDetector(W, b)


	for filename in os.listdir(folder):

		# read one test image
	
		img = cv2.imread(os.path.join(folder,filename))
		if img is None:
			break

		cv2.imshow("img",img)
		
		mask_img = my_detector.segment_image(img)
		cv2.imshow("masked img",mask_img)
		cv2.waitKey()
		cv2.destroyAllWindows()

		my_detector.get_bounding_box(img)

			
			
		

