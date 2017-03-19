"""
from samples, create 'licence plate only' jpg files
output files are named as 0.jpg, 1.jpg, ...

WORKS BEST:
cp TrainingWithITO40/classifier/cascade.xml ./rekkari.xml

python3 ../rekkariDetectionSave.py 5
detects all, but gives two rectangles, where smaller is inside the bigger

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
import sys

rekkari_cascade = cv2.CascadeClassifier('/home/mka/PycharmProjects/Rekkari/rekkari.xml')

images = glob('*.jpg')+glob('*.JPG')+glob('*.jpeg')+glob('*.JPEG')
print(images)
try:
    scale = int(sys.argv[1])
except:
    scale = 5

for i, img_name in enumerate(images):
    print(i, img_name)
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clone = img.copy()
    print("scale", scale)
    rekkaris = rekkari_cascade.detectMultiScale(img, 1.03, scale, minSize=(5,18))
    for (x,y,w,h) in rekkaris:
        print("xywh",x,y,w,h)
        cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),5)
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        cv2.imwrite(str(i)+'-'+str(x)+'.jpg', roi_gray)

    #if not(img.empty()):
    #cv2.imshow('img',clone)
    plt.imshow(clone)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
