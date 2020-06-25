## This can be used to study the frequency spectrum of a gaussian filter

import math,numpy as np
import matplotlib.pyplot as plt

def gaussianKernel(size_x, sigma):
	kernel = np.fromfunction(lambda x:(1/math.sqrt(2*math.pi*sigma**2)*(math.e**(-1*((x-(size_x-1)/2)/(2*sigma))**2))),(size_x,));kernel = (kernel - np.min(kernel))/(np.max(kernel)-np.min(kernel));return kernel

def drawArrow(axes, x,y,dx,dy, annotation):
	axes.arrow(x,y,dx ,dy ,length_includes_head=True,head_width = 0.025,head_length = 1)
	axes.arrow(x+dx,y+dy,-dx, -dy,length_includes_head=True,head_width = 0.025,head_length = 1)
	axes.annotate(annotation,xy = (x+dx/2, y+dy/2),xycoords = 'data',fontsize = 20)


def drawFrequencySpectrum(tlow, thigh):
	fig = plt.figure(figsize = (100,100))
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.xlabel("Frequency in cycle/image",fontsize = 20)
	plt.ylabel("Gain",fontsize = 20)

	x= np.arange(0,99,1)

	g1 = gaussianKernel(200,tlow)
	g2 = 1- gaussianKernel(200,thigh)
	g1 = g1[int(len(g1)/2):-1]
	g2 = g2[int(len(g2)/2):-1]
	x1=np.argmin(np.abs(0.5-g1));
	x2=np.argmin(np.abs(0.5-g2));

	axes = fig.add_subplot(111)
	axes.plot(x,g1,color = 'b')
	axes.plot(x,g2,color = 'r')

	x = min(x1,x2); dx = abs(x2-x1)
	y = min(g1[x1],g2[x2]); dy = abs(g1[x1]-g2[x2])
	drawArrow(axes,x,y,dx,dy,"Gap")
	plt.show()

drawFrequencySpectrum(tlow = 10, thigh = 20)

