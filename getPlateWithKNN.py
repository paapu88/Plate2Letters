# python3 getPlateWithKNN.py '/home/mka/PycharmProjects/Rekkari/Training/img/sample_pos5.jpg' 'flattened_images-x8-y12.txt'

import cv2
import numpy as np
import operator
import os
import sys
from myContours import ContoursWithFilters
from mySlices import MySlices


resolution=(8,12)

RESIZED_IMAGE_WIDTH = resolution[0]
RESIZED_IMAGE_HEIGHT = resolution[1]


class GetPlateWithKNN():
    """ train KNN classigfier
        use it to get the characters from a Finninsh licence plate"""

    def __init__(self, classificationFileName='classifications.txt',
                 flatImagesFileName=None):

        if flatImagesFileName is None:
            flatImagesFileName = "flattened_images-x" + str(RESIZED_IMAGE_WIDTH) + \
                                 "-y" + str(RESIZED_IMAGE_HEIGHT) + ".txt"
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
        self.image=None
        self.slices = None
        self.sliceNR = None # testing

    def setImage(self, imagefile):
        self.image = cv2.imread(imagefile, 0)
        self.sliceNR = 0

    def setSlices(self):
        self.slices = MySlices(intWidth=self.image.shape[1], intHeight=self.image.shape[0])
        print("IMAGE SIZE", self.image.shape[1], self.image.shape[0])

    def plate2CharsWithContours(self, useBlur=False):
        """ for a given image, returns the plate 5 or 6 characters as characters"""
        # read in testing plate
        contours = ContoursWithFilters(image = self.image)
        #cv2.imshow("imgT", image)
        #intChar = cv2.waitKey(0)
        # get grayscale image
        contours.setGray()
        # blur
        if useBlur:
            contours.setBlur()
        # filter image from grayscale to black and white, use threshold
        #contours.setThreshold(thres_pixel_neib=8, thres_pixel_sub=1)
        contours.setThreshold()        
        # show threshold image for reference
        #cv2.imshow("imgThresh", contours.getThreshold())
        #intChar = cv2.waitKey(0)
        # estimate the contours by opencv
        contours.setContours()
        print("len contours", contours.getLenContours())
        #for contour in contours.getContoursOnly():
        #    [intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
        #    # draw a green rect around the current char
        #    cv2.rectangle(image,
        #                  (intX, intY),  # upper left corner
        #                  (intX + intWidth, intY + intHeight),  # lower right corner
        #                  (0, 255, 0),  # green
        #                  1)  # thickness
        #cv2.imshow("imgT", image)
        #intChar = cv2.waitKey(0)

        # mark contours that have bad size or aspect ratio
        contours.checkContourArea(min_area=50, max_area=6000)
        contours.checkContourRatio(max_ratio=7, min_ratio=0.8)
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
        for contour in contours.getContoursOnly():
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
            # draw a green rect around the current char
            cv2.rectangle(self.image,
                          (intX, intY),  # upper left corner
                          (intX + intWidth, intY + intHeight),  # lower right corner
                          (0, 0, 255),  # green
                          1)  # thickness
        cv2.imshow("imgT", self.image)
        intChar = cv2.waitKey(0)

        # now we should find (5) or 6 countours of same height
        # near each other
        ypos_sets = contours.defineSets(criterium='ypos')
        print("YPOS:",len(ypos_sets))

        height_sets = contours.defineSets(criterium='height')
        print("HEIGHT:", len(height_sets))

        # take interseptions that have 5-6 members, longest come first
        ok_sets = contours.getInterception(set1=ypos_sets, set2=height_sets)
        #ok_sets = contours.getContoursOnly()
        print("LEN",len(ok_sets))

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
            # check subsequent x condition
            ok = contours.checkSubsequent()
            for contour in contours.getContoursOnly():
                [intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
                # draw a green rect around the current char
                cv2.rectangle(self.image,
                              (intX, intY),  # upper left corner
                              (intX + intWidth, intY + intHeight),  # lower right corner
                              (0, 255, 255),  # green
                              2)  # thickness
            cv2.imshow("imgT", self.image)
            intChar = cv2.waitKey(0)


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
            cv2.rectangle(image,
                          (intX, intY),     # upper left corner
                          (intX + intWidth, intY + intHeight),# lower right corner
                          (0, 0, 255),              # blue
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

    def plate2CharsWithSlides(self):
        # slices = MySlices(intWidth=self.image.shape[1],intHeight=self.image.shape[0])

        for plate in self.slices.getPlates():
            strFinalString=''
            clone = self.image.copy()
            for (intX, intY, intWidth, intHeight) in plate:
                # resize image, for recognition and storage
                imageSmall = self.image.copy()[intY : intY + intHeight, intX : intX + intWidth]
                imageSmallResized = cv2.resize(imageSmall,(RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT))
                # flatten image into 1d numpy array
                oneDimageSmallResized = imageSmallResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                oneDimageSmallResized = np.float32(oneDimageSmallResized)

                # call KNN function find_nearest
                retval, npaResults, neigh_resp, dists \
                = self.kNearest.findNearest(oneDimageSmallResized, k = 1)

                # get character from results
                strCurrentChar = str(chr(int(npaResults[0][0]))).upper()
                # append current char to full string
                strFinalString = strFinalString + strCurrentChar
                cv2.rectangle(clone,(intX, intY), (intX+intWidth,intY+intHeight),
                          (0,255,0),3) # top-left, bottom-right
                if self.sliceNR == 279:
                    cv2.imwrite(str(self.sliceNR)+str(intX)+'.test.del.jpg', imageSmallResized)

            cv2.imwrite(str(self.sliceNR)+'.test.del.jpg', clone)

            print(str(self.sliceNR)+'   ' + strFinalString)
            self.sliceNR = self.sliceNR + 1


###################################################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        getplate = GetPlateWithKNN(flatImagesFileName=sys.argv[2])

    getplate.setImage(imagefile=sys.argv[1])


    #print("PLATE RESULT with contours:", getplate.plate2CharsWithContours(useBlur=True))
    #getplate.setImage(imagefile=sys.argv[1])

    getplate.setSlices()
    getplate.plate2CharsWithSlides()

    getplate.slices.makeSmaller()
    getplate.plate2CharsWithSlides()

    getplate.slices.makeSmaller()
    getplate.plate2CharsWithSlides()

    getplate.slices.makeSmaller()
    getplate.plate2CharsWithSlides()

    getplate.slices.makeSmaller()
    getplate.plate2CharsWithSlides()

    sys.exit()
    #print("PLATE RESULT with contours:", getplate.plate2CharsWithContours(image=image, useBlur=True))
    #while True:
    #    getplate.plate2CharsWithSlides()
    #    myContinue = getplate.slices.makeSmaller()
    #    if not myContinue:
    #        break









