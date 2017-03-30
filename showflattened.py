# python3 showflattened.py flattened_images-x10-y15.txt 15 10
""" get 1d flattened image to an picture on the sreen
see CHANGE below
"""
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt


A = np.loadtxt(sys.argv[1])
#print(A.shape)
ydim=int(sys.argv[2])
xdim=int(sys.argv[3])
B=A.reshape((A.shape[0],ydim, xdim))
#print(B)


bigfig= np.concatenate(B[:][:][:],1)
print (bigfig.shape)
#newshapeY = int(round(bigfig.shape[0]*2))
#newshapeX= int(round(bigfig.shape[1]/2))
#print(newshapeY, newshapeX)
#bigfig= bigfig.reshape((150, -1))

# CHANGE (if dimensions do not match)
bigfig= np.hsplit(bigfig,5)
bigfig= np.concatenate(bigfig[:][:][:],0)

#bigfig=np.asarray(bigfig)
#bigfig = np.asarray(bigfig)
print(bigfig.shape)
plt.imshow(bigfig, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


