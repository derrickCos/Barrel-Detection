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


class BarrelDetector():
	def __init__(self, W, b):
		'''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier

			The BarrelDetector object has weights (W) and bias (b) as attributes
			These have been computed using a logistic regression model
		'''

		self.W = W
		self.b = b

	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise

			Heuristic for determining blue regions:
			    Any pixel value that is greater than mean + 2*standard_deviation 
		'''
		# YOUR CODE HERE
		img = (img - 127.5)/255.0
		mask_img = sigmoid(np.sum(img*self.W, axis = 2) + self.b)

		# Heuristic for determining blue regions

		mean = np.mean(mask_img)
		std = np.std(mask_img)

		mask_img[mask_img > mean + 2*std] = 1
		mask_img[mask_img <= mean + 2*std] = 0
		
		kernel = np.ones((0,0),np.uint8)
		mask_img = cv2.dilate(mask_img,kernel,iterations = 1)
		labeled_image = label(mask_img)
		mask_img = mask_img*0

		for region in regionprops(labeled_image):

			if region.eccentricity > 0.7 and region.area>500 and (region.orientation > np.pi/4 or region.orientation < -np.pi/4):

				minr, minc, maxr, maxc = region.bbox
				mask_img[minr:maxr,minc:maxc] = 255

		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE 

		binaryIm = self.segment_image(img)
		binaryIm = np.uint8(binaryIm*255)
		
		
		labeled_image = label(binaryIm)

		fig, ax = plt.subplots(figsize=(10, 6))
		ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


		for region in regionprops(labeled_image):

			if region.eccentricity > 0.7 and region.area>500 and (region.orientation > np.pi/4 or region.orientation < -np.pi/4):
			
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
					fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)

		ax.set_axis_off()
		plt.tight_layout()
		plt.show()

		return 0


def annotate(img):
	'''
	   Create a polygon for binary masking 
	   Call the funciton after plt.imshow(your_image)
	'''
	
	my_roi = RoiPoly(color = 'r')
	mask = my_roi.get_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	return mask


def sigmoid(x):
	'''
	   Computes the sigmoid for logistic regression model
	   Maps x to a value between (0,1)
	   Numerically stable version of sigmoid
	'''
	return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def log_reg(img, y, W, b, learning_rate = 0.1):

	# Compute Model Output 

	out = sigmoid(np.sum(img*W, axis = 2) + b)

	# Cross Entropy Loss

	loss = -1*( y*np.log(out) +  (1-y)*np.log(1-out) )
	num = loss.shape[0]*loss.shape[1]
	loss = np.sum(loss)/num

	# Compute Updates 
	error = out - y
	grad_W = img * np.expand_dims(error,2) 
	grad_W = np.sum(grad_W,(0,1))/num
	grad_b = np.sum(error)/num

	# Adjust weights and biases

	new_W = W - learning_rate*grad_W
	new_b = b - learning_rate*grad_b

	return new_W, new_b, loss



if __name__ == '__main__':
	
	# Choose Trainset/Validation Set (../trainset or ../validation)

	folder = "../trainset" 


	create_labels = False

	train = False

	if create_labels:

		for filename in os.listdir(folder):

			# read one test image
		
			img = cv2.imread(os.path.join(folder,filename))

			# Annotation Process

			plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			mask1 = annotate(img)
			cv2.waitKey(0)

			plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			mask2 = annotate(img)
			mask = mask1 + mask2

			plt.imsave(os.path.join(folder,"labels",filename),mask, cmap = "gray")

			cv2.waitKey(0)
			cv2.destroyAllWindows()


	# Pixel-wise Logistic Regression Model

	# Weight and Bias Initialization 

	W = np.random.normal(size = (1,3))
	b = np.random.normal(size = (1,1))
	W = np.load("weights1.npy")
	b = np.load("bias1.npy")


	if train:
		epochs = 50

		for i in range(epochs):

			epoch_loss = 0

			for filename in os.listdir(folder):

				# read one test image
			
				img = cv2.imread(os.path.join(folder,filename))
				
				# Normalize Image

				if img is None:
					break

				img = (img - 127.5)/255.0

				# Get labels

				y = plt.imread(os.path.join(folder,"labels",filename))
				y = y[:,:,0]

				# Update weight and bias and compute Cross-Entropy loss

				W, b, loss = log_reg(img, y, W, b, 0.5)

				epoch_loss += loss


			print(epoch_loss)

			# Save weights and biases

			np.save("weights1", W)
			np.save("bias1", b)

	# Test Model

	else:

		W = np.load("weights1.npy")
		b = np.load("bias1.npy")
		my_detector = BarrelDetector(W, b)

		for filename in os.listdir(folder):

			# read one test image
		
			img = cv2.imread(os.path.join(folder,filename))
			if img is None:
				break

			my_detector.get_bounding_box(img)
