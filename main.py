import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import scipy
import pandas as pd
import seaborn as sn

def gaussianKernel(size_x, size_y, sigma):
	kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size_x-1)/2)**2+(y-(size_y-1)/2)**2))/(2*sigma**2)), (size_x, size_y))
	return (kernel-np.min(kernel)) / (np.max(kernel)-np.min(kernel))

image = cv2.imread('dog.jpeg')     
image = cv2.resize(image, (200,200), interpolation = cv2.INTER_CUBIC)
image2 = cv2.imread('eagle.jpeg')
image2 = cv2.resize(image2, (200,200), interpolation = cv2.INTER_CUBIC)

images = np.concatenate((image,image2),axis = 1)
cv2.imshow('images',images)
cv2.waitKey(0)
cv2.destroyAllWindows()

def channel():					# arguments are +/- percentages shift
	full = np.empty(image.shape)
	GaussianPyr = [] 									# 3 channels
	LaplacianPyr = []
	PyrSize = 8

	for i in range(PyrSize):
		GaussianPyr.append([])	
		LaplacianPyr.append([])	

	for i in range(3):
		img = image[:,:,i]
		img2 = image2[:,:,i]		
		rows, cols = img.shape
		threshold_low = 5
		threshold_high = 30 
		
		img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
		img_shift = np.fft.fftshift(img_dft)
		img2_dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
		img2_shift = np.fft.fftshift(img2_dft)
		
		mask_low = np.zeros((rows,cols,2))
		mask_high = np.zeros((rows,cols,2))
		mask_low[:,:,0] = gaussianKernel(rows, cols, threshold_low)
		mask_low[:,:,1] = np.copy(mask_low[:,:,0])
		mask_high[:,:,0] = 1 - gaussianKernel(rows, cols, threshold_high)
		mask_high[:,:,1] = np.copy(mask_high[:,:,0])

		tot_dft = img_shift*mask_low + img2_shift*mask_high
		tot_mag = cv2.magnitude(tot_dft[:,:,0], tot_dft[:,:,1]) 
		tot_log = 20 * np.log(tot_mag+1)
		tot_ishift = np.fft.ifftshift(tot_dft)
		tot_back = cv2.idft(tot_ishift)
		tot_inv = cv2.magnitude(tot_back[:,:,0],tot_back[:,:,1])
		full[:,:,i] = tot_inv        

		G = tot_inv
		for j in range(PyrSize):
			GaussianPyr[j].append(G)
			G = cv2.pyrDown(G) 

		for j in range(PyrSize-1, 0, -1):
		    GE = cv2.pyrUp(GaussianPyr[j][i])
		    tupShape = GaussianPyr[j-1][i].shape
		    GE = cv2.resize(GE, tupShape, interpolation = cv2.INTER_CUBIC)
		    L = cv2.subtract(GaussianPyr[j-1][i], GE)
		    LaplacianPyr[j-1].append(L)

	#--------------------------------------------------------------------------------
	full /= np.max(full)
	cv2.imshow('hybrid',full)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('hybrid.jpeg', np.int32(full*255))
	#--------------------------------------------------------------------------------
	# generate Gaussian pyramid for Hybrid image

	GaussianList = []	
	for i in range(PyrSize):
		tupSize = GaussianPyr[i][0].shape
		fullImage = np.empty((tupSize[0], tupSize[1], 3))
		for j in range(3):
			fullImage[:,:,j] = GaussianPyr[i][j]

		fullImage /= np.max(fullImage)
		fullImage = cv2.resize(fullImage, (200,200), interpolation = cv2.INTER_CUBIC)
		GaussianList.append(fullImage)

	imageTuple = tuple(GaussianList)
	images = np.concatenate(imageTuple, axis = 1)

	cv2.imshow('Gaussian', images)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('GaussianPyr.jpeg', np.int32(images*255))
	#--------------------------------------------------------------------
	# generate Laplacian pyramid for Hybrid image

	LaplacianList = []	
	for i in range(PyrSize - 1):
		tupSize = LaplacianPyr[i][0].shape
		fullImage = np.empty((tupSize[0], tupSize[1], 3))
		for j in range(3):
			fullImage[:,:,j] = LaplacianPyr[i][j]

		fullImage /= np.max(fullImage)
		fullImage = cv2.resize(fullImage, (200,200), interpolation = cv2.INTER_CUBIC)
		LaplacianList.append(fullImage)

	imageTuple = tuple(LaplacianList)
	images = np.concatenate(imageTuple, axis = 1)

	cv2.imshow('Laplacian', images)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('LaplacianPyr.jpeg', np.int32(images*255))
	#--------------------------------------------------------------------
	# generate Correlation matrix for Laplacian images

	data = {}
	cols = []

	for i in range(PyrSize - 1):
		col = 'L' + str(i)
		data[col] = LaplacianList[i].flatten()
		cols.append(col)

	df = pd.DataFrame(data, columns=cols)

	corrMatrix = df.corr()
	print (corrMatrix)
	fig = sn.heatmap(corrMatrix, annot=True).get_figure()
	fig.show()
	fig.savefig('CorrelationMatrix.png')

#--------------------------------------------------------------------
channel()