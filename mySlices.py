# the resolution should be the same as in
# GenData (there it is 2/3, here 5/7...)

import numpy as np
import sys

class MySlices():
    """
    This is brute force way to get boxes where to search characters
    more clever might be to use histogram of intensities to get
    borders between characters and top+bottom

    INPUT opencv rectangle
    (intX, intY),  # upper left corner
    (intX + intWidth, intY + intHeight),  # lower right corner

    OUTPUT: np array of plates
    each plate consists of 7 rectangles for character recognition and 8'th item which
    has [centerX centerY 0 0], where the canter of the plate is given

    If initial reactangles do not result character regocnition use
    makeSmaller
    or
    makeBigger

    to get smaller/bigger rectangles for character recognition

    this works only for Finnish plates where we have "-" in the middle
    (that one has only 17/44 width compare to normal alphabet

    """

    
    def __init__(self, intX=0, intY=0, intWidth=None, intHeight=None, strideX=2, strideY=2,
                 XYratio=2/3, plateWidth=6.38, middleWidth=0.38, minHeight=6, nMakeSmallerSteps=10):
        self.intX = intX           # upper left (usually zero, if not add in the end, NOT DONE)
        self.intY = intY           # upper left (usually zero, if not add in the end, NOT DONE)
        self.intWidth = intWidth   # initial guess of the width of the plate
        self.intHeight = intHeight # initila guess of the height of the plate
        self.strideX = strideX     # how much grid moves right in pixel
        self.strideY = strideY     # how much grid moves down in pixel
        self.height = intHeight    # current height of on character
        self.XYratio = XYratio     # character x/y ratio
        self.plateWidth = plateWidth # width of the exected character region in terms of one character width
        self.middleWidth = middleWidth  # how much central "-" is in one-character width
        self.minHeight = minHeight   # the minimim width of an character in pixels, return None when this is met
        self.nMakeSmallerSteps = nMakeSmallerSteps  # how many steps from plate hight to min character hight
        self.setInitialWidthHeight()
        self.makeSmallerStep = int(round((self.intHeight - self.minHeight)/self.nMakeSmallerSteps))


    def setInitialWidthHeight(self):
        """ set initial (maximal) width and height for one-character region"""
        self.width = int(round(self.height * self.XYratio))
        # check that 7x width is not bigger than plate
        while self.plateWidth * self.width > self.intWidth:
            self.width = self.width - 1
        self.height = int(round(self.width / self.XYratio))
        print("WIDTH HEIGHT",self.width, self.height)

    def makeSmaller(self, absoluteStep=None):
        """ make character box smaller, step = ratio * original image"""
        if absoluteStep:
            self.height = self.height - absoluteStep
            print ("SH", self.height)
        else:
            self.height = self.height - self.makeSmallerStep
        self.width = int(round(self.height * self.XYratio))
        if self.height > self.minHeight:
            return True
        else:
            return False

    def getPlateHeight(self):
        return self.intHeight


    def getPlates(self):
        """ With current one-character width and height and with current srtideX, strideY
        OUTPUT all possible one-character boxes"""
        currentTopLeftX = 0
        currentTopLeftY = 0
        left = 0
        top = 0
        # single box widths
        currentMiddle = int(round(self.middleWidth * self.width))
        addX = [self.width, self.width, self.width, currentMiddle,
                self.width, self.width, self.width]
        # x position of character in single plate in terms of character width
        sumX = [0, 1 * self.width, 2 * self.width,
                3* self.width,
                3 * self.width + currentMiddle,
                4 * self.width + currentMiddle,
                5 * self.width + currentMiddle]

        #print(sumX, addX)

        plates = []
        while (currentTopLeftY < self.height ):  # stride Y, only first one character needed for stride
            while (currentTopLeftX  < self.width): # stride X, only first character needed
                while (currentTopLeftY+top + self.height) < self.intHeight: #height Y
                    while (currentTopLeftX+left + self.width*self.plateWidth) < self.intWidth:  #3+1+3 characters in x
                        boxes = []
                        for charPos, charWidth in zip(sumX, addX):
                            boxes.append([currentTopLeftX+left+charPos,
                                          currentTopLeftY+top,
                                          charWidth,
                                          self.height])
                            #print("X:",currentTopLeftX+left+charPos)
                        #centerX = int(round(boxes[3][0]+0.5*boxes[3][2]))
                        #centerY = int(round(boxes[3][1]+0.5*boxes[3][3]))
                        #boxes.append([centerX, centerY, 0, 0])  # two last zeros are dummies at the moment
                        plates.append(boxes)
                        left = left + self.plateWidth
                    top = top + self.height
                    left = 0
                currentTopLeftX = currentTopLeftX + self.strideX
                top = 0
                left = 0
            currentTopLeftY = currentTopLeftY + self.strideY
            currentTopLeftX = 0
            top = 0
            left = 0

        return np.asarray(plates, dtype=np.uint16)

import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])

    plates = MySlices(intWidth=img.shape[1],intHeight=img.shape[0])
    # print(plates.getPlates().shape)

    plates.makeSmaller()
    plates.makeSmaller()
    plates.makeSmaller()
    #print("BOX SHAPE",boxes.getBoxes().shape)
    #print
    # print("SHAPE:", img.shape)
    clone = img.copy()
    plt.imshow(clone, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    while True:
        izero = 0
        for plates in plates.getPlates():
         for rectangle in plates:

            # print(rectangle[0], rectangle[1], rectangle[2], rectangle[3])
            #plt.gca().add_patch(plt.Rectangle((rectangle[0], boxes.getPlateHeight()-rectangle[1]),
            #                                 rectangle[2]-rectangle[0],
            #                                 rectangle[3]-rectangle[1],
            #                                 color='r', fill=True))
            cv2.rectangle(clone,(rectangle[0],
                               rectangle[1]),
                          (rectangle[0]+rectangle[2],
                          rectangle[1]+rectangle[3]),
                          (0,255,0),3) # top-left, bottom-right
            izero = izero+1
            if izero> 6:
                izero=0
                plt.pause(1)

                plt.gcf().clear()
                clone = img.copy()
                plt.imshow(clone, cmap='gray', interpolation='bicubic')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


        #plt.clf()
        myContinue = plates.makeSmaller()
        if not myContinue:
            break

    plt.show()