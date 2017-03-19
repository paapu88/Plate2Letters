# python3 TrainAndTest.py '~/PycharmProjects/Rekkari/Training/img/sample_pos5.jpg'

import cv2
import numpy as np
import operator
import os
import sys
from myContours import ContoursWithFilters

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main(*args):
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program")
        os.system("pause")
        return
    # end try

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications \
        = npaClassifications.reshape((npaClassifications.size, 1))       
    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()                   
    # train it
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    # to my understanting, in opencv3 one cannot save the trained classifier

    # read in testing plate
    contours = ContoursWithFilters(image =  cv2.imread(sys.argv[1]))
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
    #print("len contours", contours.getLenContours())
    # mark contours that have bad size or aspect ratio
    contours.markTooSmallorWrongAspectRatio()
    #print("kill smalls:", contours.getKillIndex())
    # set contours whose parent has been killed to be orphans
    contours.setOrphans()
    # mark all contours who still have parents for removal
    # (some stuff inside an alphabet)
    contours.markContoursWithParents()
    # finally kill all marked contours
    # (wrong shape or having a parent)
    contours.removeMarkedContours()
    print("len contours 2 : ", contours.getLenContours())
    # now we should find (5) or 6 countours of same height
    # near each other
    ypos_sets = contours.defineSets(criterium='ypos')
    height_sets = contours.defineSets(criterium='height')
    # take interseptions that have 5-6 members
    ok_sets6 = []
    ok_sets5 = []    
    for ypos_set in ypos_sets:
        for height_set in height_sets:
            myset = ypos_set.intersection(height_set)
            if len(myset)==6 and myset not in ok_sets6:
                ok_sets6.append(myset)
            elif len(myset)==5 and myset not in ok_sets6:
                ok_sets5.append(myset)
    # take first 6 member set if exists, otherwise take first 5 member set
    # otherwise fail
    
    print("len 6 sets, 5 sets: ", len(ok_sets6), len(ok_sets5))
    print(ok_sets6, ok_sets5)
    if len(ok_sets6) > 0:
        ok_set = ok_sets6[0]
    elif len(ok_sets5) > 0:
        ok_set = ok_sets5[0]
    else:
        print("ypos_sets:", ypos_sets)
        print("height_sets:", height_sets)        
        raise notImplementedError("PLATE LETTERS NOT FOUND")
    # mark all contours not belonging the found set for removal
    contours.markNonSetContours(myset = ok_set)
    # remove sets not belonging to letters in the number plate
    contours.removeMarkedContours()
    # sort contours by x coord
    contours.sortContours(mykey='x')

    # declare final string,
    # this will have the final number sequence by the end of the program
    strFinalString = ""         

    image = contours.getImage()
    for contour in contours.getContoursOnly():
        [intX, intY, intWidth, intHeight]=cv2.boundingRect(contour)         
        # draw a green rect around the current char
        cv2.rectangle(image,
                      (intX, intY),     # upper left corner
                      (intX + intWidth, intY + intHeight),# lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        # crop char out of threshold image
        imgROI = contours.getThreshold()[intY : intY + intHeight,     
                                         intX : intX + intWidth]

        # resize image, for recognition and storage
        imgROIResized = cv2.resize(imgROI,
                                   (RESIZED_IMAGE_WIDTH,
                                    RESIZED_IMAGE_HEIGHT))
        
        # flatten image into 1d numpy array
        npaROIResized \
            = imgROIResized.reshape\
                ((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        
        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)       

        # call KNN function find_nearest
        retval, npaResults, neigh_resp, dists \
            = kNearest.findNearest(npaROIResized, k = 1)     

        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0]))).upper()                                             
        # append current char to full string
        strFinalString = strFinalString + strCurrentChar            

    #add the '-' sign
    strWithMinus = strFinalString[:-3] + '-' + strFinalString[-3:]
    # show the full string    
    print("PLATE:  " + strWithMinus)                  

    # show input image with green boxes drawn around found digits
    cv2.imshow("imgTestingNumbers", image)
    # wait for user key press
    cv2.waitKey(0)                                          

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main(sys.argv)
# end if









