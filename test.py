# python3 ./test.py "./Positives/*jpg"

from glob import glob
import os, sys

files = glob(sys.argv[1])

for file in files:
    command = 'python3 TrainAndTest.py '+file
    os.system(command)
    
