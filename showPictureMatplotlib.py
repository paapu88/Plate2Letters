# python3 ../showPictureMatplotlib.py pos10.jpg

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1])
plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# T: x 36, y 57
# 9 x 38 y 57
# ratio is near 1.5
