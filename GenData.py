# python3 GenData.py "/home/mka/Pictures/rekkariKirjaimistoMusta.png"
# python3 GenData.py "/home/mka/PycharmProjects/OpenCV_3_KNN_Character_Recognition_Python/Positives/*5.jpg"

import sys
import numpy as np
import cv2
import os
from myContours import ContoursWithFilters

# the resolution should be the same as in
# mySlices (there it is 5/7, here 2/3...)

resolutions = [(8,12),(10,15)]
npaFlattenedImagesWithRes = [None, None]
#npaFlattenedImagesWithRes.append(np.empty(0, resolutions[0][0]*resolutions[0][1]))
#npaFlattenedImagesWithRes.append(np.empty(0, resolutions[1][0]*resolutions[1][1]))


def main(*args):
    from glob import glob


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
                     ord('y'), ord('z'), ord('å'),
                     ord('ä'), ord('ö') ]

    files = glob(sys.argv[1])
    print("files:", files)
    for file in files:
        print("file:", file)
        # read in training image
        contours = ContoursWithFilters(image =  cv2.imread(file))
        # get grayscale image
        contours.setGray()
        # blur
        contours.setBlur(blurSize=(7,7))
        # filter image from grayscale to black and white, use threshold
        contours.setThreshold(thres_pixel_neib=7, thres_pixel_sub=2)
        # show threshold image for reference
        #cv2.imshow("imgThresh", contours.getThreshold())
        #intChar = cv2.waitKey(0)
        # estimate the contours by opencv
        contours.setContours()
        print("len contours 1", contours.getLenContours())
        # mark contours that have bad size or aspect ratio
        contours.checkContourArea(min_area=1000, max_area=6000)
        contours.checkContourRatio(max_ratio=7, min_ratio=0.8)
        # set contours whose parent has been killed to be orphans
        contours.setOrphans()
        # mark all contours who still have parents for removal
        # (some stuff inside an alphabet)
        contours.markContoursWithParents()
        # finally kill all marked contours
        # (wrong shape or having a parent)
        contours.removeMarkedContours()
        print("len contours 2", contours.getLenContours())
        ok_contours = contours.getContoursOnly()

        imgCopy = contours.getImage()
        for npaContour in ok_contours:
            # get and break out bounding rect
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgCopy,   #original (color?)image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            # crop char out of threshold image
            #imgROI = contours.getThreshold()[intY:intY+intH, intX:intX+intW]

            # show cropped out char for reference
            # cv2.imshow("imgROI", imgROI)
            # show training numbers image,
            # this will now have red rectangles drawn on it
        cv2.imshow("training", imgCopy)
        intChar = cv2.waitKey(0)                     # get key press

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

            # show cropped out char for reference
            # cv2.imshow("imgROI", imgROI)
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

                for i, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT) in enumerate(resolutions):

                    # resize image
                    imgROIResized \
                        = cv2.resize(imgROI,
                                     (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

                    # show resized image for reference
                    cv2.imshow("imgROIResized", imgROIResized)
                    # flatten image to 1d numpy array so we can write to file later
                    npaFlattenedImage \
                        = imgROIResized.\
                        reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    npaFlattenedImage = np.float32(npaFlattenedImage)
                    # add current flattened impage numpy array to
                    # list of flattened image numpy arrays
                    if npaFlattenedImagesWithRes[i] is None:
                        npaFlattenedImagesWithRes[i] = npaFlattenedImage
                    else:
                        npaFlattenedImagesWithRes[i] = np.append(npaFlattenedImagesWithRes[i], npaFlattenedImage, 0)


    for i, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT) in enumerate(resolutions):
        # convert classifications list of ints to numpy array of floats
        fltClassifications = np.array(intClassifications, np.float32)
        # flatten numpy array of floats to 1d so we can write to file later
        npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

        print("\n\ntraining complete !!\n")

        np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
        np.savetxt("flattened_images-x"+str(RESIZED_IMAGE_WIDTH)+"-y"+\
            str(RESIZED_IMAGE_HEIGHT)+".txt", npaFlattenedImagesWithRes[i])

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




