# python3 TrainAndTest.py '~/PycharmProjects/Rekkari/Training/img/sample_pos5.jpg'

import cv2
import numpy as np
import operator
import os
import sys
from myContours import ContoursWithFilters

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


class GetPlateWithKNN():
    """ train KNN classigfier
        use it to get the characters from a Finninsh licence plate"""

    def __init__(self, classificationFileName='classifications.txt',
                 flatImagesFileName='flattened_images.txt'):
        # train the KNN classifier, unfortunately a trained classifier cannot be saved/loaded
        # read in training classifications
        print("Training classifier start")
        npaClassifications = np.loadtxt(classificationFileName, np.float32)
        # read in training images
        npaFlattenedImages = np.loadtxt(flatImagesFileName, np.float32)
        # reshape numpy array to 1d, necessary to pass to call to train
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        # instantiate KNN object
        self.kNearest = cv2.ml.KNearest_create()
        # train it
        self.kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        # to my understanting (mok), in opencv3 one cannot save the trained classifier
        print("Training classifier end")

    def plate2Chars(self, image=None):
        """ for a given image, returns the plate 5 or 6 characters as characters"""
        # read in testing plate
        contours = ContoursWithFilters(image = image)
        #cv2.imshow("imgT", image)
        #intChar = cv2.waitKey(0)
        # get grayscale image
        contours.setGray()
        # blur
        contours.setBlur()
        # filter image from grayscale to black and white, use threshold
        #contours.setThreshold(thres_pixel_neib=8, thres_pixel_sub=1)
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
        # print("len contours 2 : ", contours.getLenContours())
        # now we should find (5) or 6 countours of same height
        # near each other
        ypos_sets = contours.defineSets(criterium='ypos')
        height_sets = contours.defineSets(criterium='height')

        # take interseptions that have 5-6 members, longest come first
        ok_sets = contours.getInterception(set1=ypos_sets, set2=height_sets)

        #loop possible contours, until subsequent x criterium is fulfilled
        ok = False
        for ok_set in ok_sets:
            clone = contours.getContoursOnly()
            # mark all contours not belonging the found set for removal
            #print("okset", ok_set)
            contours.markNonSetContours(myset = ok_set)
            # remove sets not belonging to letters in the number plate
            contours.removeMarkedContours()
            # sort contours by x coord
            contours.sortContours(mykey='x')
            #    check subsequent x condition
            ok = contours.checkSubsequent()
            if ok:
                break
            contours.setContoursOnly(npaContours=clone)

        if not ok:
            return 'NOT FOUND'
            #raise RuntimeError("could not find letters, sorry")
        # declare final string,
        # this will have the final number sequence by the end of the program
        strFinalString = ""

        image = contours.getImage()
        for contour in contours.getContoursOnly():
            [intX, intY, intWidth, intHeight]=cv2.boundingRect(contour)
            # draw a green rect around the current char
            #cv2.rectangle(image,
            #              (intX, intY),     # upper left corner
            #              (intX + intWidth, intY + intHeight),# lower right corner
            #              (0, 255, 0),              # green
            #             2)                        # thickness

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
                = self.kNearest.findNearest(npaROIResized, k = 1)

            # get character from results
            strCurrentChar = str(chr(int(npaResults[0][0]))).upper()
            # append current char to full string
            strFinalString = strFinalString + strCurrentChar

        #add the '-' sign
        strWithMinus = strFinalString[:-3] + '-' + strFinalString[-3:]
        # show the full string
        print("PLATE:  " + strWithMinus)

        # show input image with green boxes drawn around found digits
        #cv2.imshow("imgTestingNumbers", image)
        #    wait for user key press
        #cv2.waitKey(0)

        #cv2.destroyAllWindows()             # remove windows from memory

        return strWithMinus

###################################################################################################
if __name__ == "__main__":
    getplate = GetPlateWithKNN()
    print("PLATE RESULT:", getplate.plate2Chars(filename=sys.argv[1]))










