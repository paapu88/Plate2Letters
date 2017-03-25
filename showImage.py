# python3 ./showImage.py "/home/mka/Videos/026.jpg"

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from getPlateWithKNN import GetPlateWithKNN

img = cv2.imread(sys.argv[1])


rekkari_cascade = cv2.CascadeClassifier('rekkari.xml')
getplate = GetPlateWithKNN(flatImagesFileName='flattened_images-x10-y15.txt')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rekkaris = rekkari_cascade.detectMultiScale(gray, 1.03, 5, minSize=(37,10))
plate=''
for (x,y,w,h) in rekkaris:
    print("W, H", w, h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    plate = plate + getplate.plate2Chars(img[y:y+h, x:x+w]) + ' '


print ("plate: ", plate)
plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# T: x 36, y 57
# 9 x 38 y 57
# ratio is near 1.5
