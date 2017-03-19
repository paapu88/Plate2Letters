# python3 GenData.py "./Positives/*5.jpg"

import sys
import numpy as np
import cv2
import os
from myContours import ContoursWithFilters

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main(*args):
    from glob import glob

    npaFlattenedImages \
        =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # declare empty classifications list,
    # this will be our list of how we are classifying our
    # chars from user input, we will write to file at the end
    intClassifications = []         

    # possible chars we are interested in are digits 0 through 9,
    # put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'),
                     ord('3'), ord('4'), ord('5'),
                     ord('6'), ord('7'), ord('8'),
                     ord('9'),
                     ord('a'), ord('b'), ord('c'),
                     ord('d'), ord('e'), ord('f'),
                     ord('g'), ord('h'), ord('i'),
                     ord('j'), ord('k'), ord('l'),
                     ord('m'), ord('n'), ord('o'),
                     ord('p'), ord('q'), ord('r'),
                     ord('s'), ord('t'), ord('u'),
                     ord('v'), ord('w'), ord('x'),
                     ord('y'), ord('z'), ord('å')]

    files = glob(sys.argv[1])
    print("files:", files)
    for file in files:
        print("file:", file)
        # read in training image
        contours = ContoursWithFilters(image =  cv2.imread(file))
        # get grayscale image
        contours.setGray()
        # blur
        contours.setBlur()
        # filter image from grayscale to black and white, use threshold
        contours.setThreshold()
        # show threshold image for reference
        #cv2.imshow("imgThresh", contours.getThreshold())
        #intChar = cv2.waitKey(0) 
        # estimate the contours by opencv
        contours.setContours()
        # mark contours that have bad size or aspect ratio
        contours.markTooSmallorWrongAspectRatio()
        # set contours whose parent has been killed to be orphans
        contours.setOrphans()
        # mark all contours who still have parents for removal
        # (some stuff inside an alphabet)
        contours.markContoursWithParents()
        # finally kill all marked contours
        # (wrong shape or having a parent)
        contours.removeMarkedContours()
        
        ok_contours = contours.getContoursOnly()
        for npaContour in ok_contours:
            # get and break out bounding rect
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # draw rectangle around each contour as we ask user for input
            imgCopy = contours.getImage()
            cv2.rectangle(imgCopy,   #original (color?)image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            # crop char out of threshold image
            imgROI = contours.getThreshold()[intY:intY+intH, intX:intX+intW]
            # resize image    
            imgROIResized \
                = cv2.resize(imgROI,
                             (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     
            # show cropped out char for reference
            # cv2.imshow("imgROI", imgROI)
            # show resized image for reference
            #cv2.imshow("imgROIResized", imgROIResized)
            # show training numbers image,
            # this will now have red rectangles drawn on it            
            cv2.imshow("training", imgCopy)  

            intChar = cv2.waitKey(0)                     # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .
                print(intChar)
                # append classification char to integer list of chars
                # (we will convert to float later before writing to file)
                intClassifications.append(intChar)                                                

                # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImage \
                    = imgROIResized.\
                    reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # add current flattened impage numpy array to
                # list of flattened image numpy arrays
                npaFlattenedImages\

    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




