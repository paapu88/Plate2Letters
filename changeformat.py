# python3 changeformat "./flattened_images-x8-y12.txt"

import numpy as np
import os, sys

filein = sys.argv[1]
fileout = sys.argv[1]+'Other_FORMAT'

indata = np.loadtxt(filein)
outdata = np.array(indata, np.float32)
np.savetxt(fileout, outdata)


