# python3 GenData4Slices.py 66 99
# assuming above 2/3=x/y ratio
# in a directory containing initially only jpg, png, tiff files
# the arguments are initial box size in x and y
"""
1) Read selected picture (automated)
2) the dimension of x is forced by fixed ratio
3) you can move the rectancle by up=w down=z left=a right=d, bigger=+, smaller=-
4) when done press L if mistakes press r and start picture again
5) give corresponding alphabet by a keystroke

This generates
1) 1d array of key characters as afile
2) 1d array of correspoinding images



"""

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

resolutions = [(8,12),(10,15)]


class MouseRectangle():
    """ get a rectangle by mouse"""
    def __init__(self, xpixel=None, ypixel=None):
        #super().__init__()
        self.refPts = []
        self.oldPts = []
        self.cropping = False
        self.image = None
        self.ratio = None
        self.xpixel= xpixel
        self.ypixel = ypixel

    def set_image(self, image):
        self.image = image

    def set_init_position(self):
        upleft = [int(self.image.shape[1] / 2), int(self.image.shape[0] / 2)]
        downright = [int(self.image.shape[1] / 2) + self.xpixel, int(self.image.shape[0] / 2) + self.ypixel]
        self.set_refPts([upleft, downright])

    def reset(self):
        self.refPts = []
        self.cropping = False
        self.set_init_position()

    def get_refPts(self):
        return self.refPts

    def set_refPts(self, refPts):
        self.refPts = refPts

    def set_oldPts(self):
        for refPt in self.refPts:
            self.oldPts.append(refPt.copy())

    def get_oldPts(self):
        return self.oldPts

    def set_ratio(self, ratio):
        """self y/x ratio for the selection box"""
        if self.ratio is not None:
            self.ratio = ratio
        #else:
        #    # assume long eu plates
        #    self.ratio = 442/118

    def get_ratio(self):
        return self.ratio




class Example():


    def __init__(self, *args):

        try:
            xpixel = int(sys.argv[1])
            ypixel = int(sys.argv[2])
        except:
            print("setting default values for pixels")
            xpixel = 40
            ypixel = 10
        print("INIT:", xpixel, ypixel)

        self.mouse = MouseRectangle(xpixel=xpixel, ypixel=ypixel)
        self.mouse.set_ratio(xpixel/ypixel)
        self.intClassifications = []
        self.flattenedImages = [None for i in range(len(resolutions))]



    def getNewName(self, oldname, subdir='img'):
        """
        get new filename with extra path
        """
        import os
        #dir = os.path.dirname(oldname)
        dir = os.getcwd()
        name = os.path.basename(oldname)
        #generate new dir if it doesnot exist
        newdir = dir + '/'+ subdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        return newdir+'/'+'sample_'+name

    def saveImage(self, cropped):
        import os.path
        intValidChars = [ord('0'), ord('1'), ord('2'),
                         ord('3'), ord('4'), ord('5'),
                         ord('6'), ord('7'), ord('8'),
                         ord('9'),
                         ord('a'), ord('b'), ord('c'),
                         ord('d'), ord('e'), ord('f'),
                         ord('g'), ord('h'), ord('i'),
                         ord('j'), ord('k'), ord('l'),
                         ord('m'), ord('n'), ord('-'),
                         ord('o'),
                         ord('p'), ord('q'), ord('r'),
                         ord('s'), ord('t'), ord('u'),
                         ord('v'), ord('w'), ord('x'),
                         ord('y'), ord('z'), ord('å'),
                         ord('ä'), ord('ö')]

        print("give the letter to be saved ")
        intChar = cv2.waitKey(0)
        #intChar=input()[0]
        print(intChar)
        clone=cropped.copy()
        if (intChar in intValidChars) or (intChar in [196, 214]):  # ä and ö included
            # all samples will get individual filename
            # resize image
            self.intClassifications.append(intChar)
            for i, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT) in enumerate(resolutions):
                resized = cv2.resize(clone.copy(),(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                flattenedImage = resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                flattenedImage = np.float32(flattenedImage)
                # add current flattened impage numpy array to
                # list of flattened image numpy arrays
                if self.flattenedImages[i] is None:
                    self.flattenedImages[i] = flattenedImage
                else:
                    self.flattenedImages[i] = np.append(self.flattenedImages[i], flattenedImage,0)
                np.savetxt("flattened_images-x"+str(RESIZED_IMAGE_WIDTH)+"-y"+\
                str(RESIZED_IMAGE_HEIGHT)+".txt", self.flattenedImages[i])

                #plt.imshow(resized, cmap='gray', interpolation='bicubic')
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
            # write character codes
            # convert classifications list of ints to numpy array of floats
            fClassifications = np.array(self.intClassifications, np.float32)
            # flatten numpy array of floats to 1d so we can write to file later
            np.savetxt("classifications.txt", fClassifications.reshape((fClassifications.size, 1)))
            print("saved character"+chr(int(fClassifications[-1])))

        else:
            print ("sorry, not a valid character")


    def showDialog(self):
        import glob
        import math
        fnames =glob.glob('*jpg')
        fnames = fnames + glob.glob('*png')
        fnames = fnames + glob.glob('*tif')

        for fname in fnames:
                img = cv2.imread(fname)
                gray = cv2.imread(fname, 0)
                print("size:", img.shape[0], img.shape[1])

                clone = img.copy()
                self.mouse.set_image(image=img)
                self.mouse.set_init_position()
                refPts = self.mouse.get_refPts()

                for i in range(0, len(refPts), 2):
                    print("printing",refPts[0], refPts[1])
                    cv2.rectangle(img, tuple(refPts[i]), tuple(refPts[i + 1]), (255, 0, 0), 10)
                oldPts = self.mouse.get_oldPts()
                for i in range(0, len(oldPts), 2):
                    cv2.rectangle(img, tuple(oldPts[i]), tuple(oldPts[i + 1]), (0, 255, 0), 1)

                # keep looping until the 'q' key is pressed
                while True:
                    while True:
                        # display the image and wait for a keypress
                        cv2.imshow("image", img)
                        key = cv2.waitKey(33)
                        change = False
                        # if the 'r' key is pressed, reset the cropping region
                        if key == ord("r"):
                            img = clone.copy()
                            self.mouse.reset()
                            self.mouse.set_image(image=img)

                        elif key == ord("+"):
                            refPts = self.mouse.get_refPts()
                            refPts[-2][0]= refPts[-2][0] - 1
                            refPts[-2][1]= refPts[-2][1] - 1
                            refPts[-1][0] = refPts[-1][0] + 1
                            refPts[-1][1] = refPts[-1][1] + 1
                            self.mouse.set_refPts(refPts)
                            change = True
                        elif key == ord("-"):
                            refPts = self.mouse.get_refPts()
                            refPts[-2][0] = refPts[-2][0] + 1
                            refPts[-2][1] = refPts[-2][1] + 1
                            refPts[-1][0] = refPts[-1][0] - 1
                            refPts[-1][1] = refPts[-1][1] - 1
                            self.mouse.set_refPts(refPts)
                            change = True
                        elif key == ord("w"):
                            #up
                            refPts = self.mouse.get_refPts()
                            refPts[-2][1] = refPts[-2][1] - 1
                            refPts[-1][1] = refPts[-1][1] - 1
                            self.mouse.set_refPts(refPts)
                            change = True
                        elif key == ord("z"):
                            #up
                            refPts = self.mouse.get_refPts()
                            refPts[-2][1] = refPts[-2][1] + 1
                            refPts[-1][1] = refPts[-1][1] + 1
                            self.mouse.set_refPts(refPts)
                            change = True
                        elif key == ord("d"):
                            #up
                            refPts = self.mouse.get_refPts()
                            refPts[-2][0] = refPts[-2][0] + 1
                            refPts[-1][0] = refPts[-1][0] + 1
                            self.mouse.set_refPts(refPts)
                            change = True
                        elif key == ord("a"):
                            #up
                            refPts = self.mouse.get_refPts()
                            refPts[-2][0] = refPts[-2][0] - 1
                            refPts[-1][0] = refPts[-1][0] - 1
                            self.mouse.set_refPts(refPts)
                            change = True

                        # if the 'L' key is pressed, break from the loop
                        elif key == ord("l"):

                            print('L pressed, next give the character')
                            break
                        #print(key)
                        if change:
                            change = False
                            img = clone.copy()
                            for i in range(0,len(refPts),2):
                                cv2.rectangle(img, tuple(refPts[i]), tuple(refPts[i+1]), (255, 0, 0), 2)
                            oldPts = self.mouse.get_oldPts()
                            for i in range(0, len(oldPts), 2):
                                cv2.rectangle(img, tuple(oldPts[i]), tuple(oldPts[i + 1]), (0, 255, 0), 1)
                            self.mouse.set_image(image=img)

                    # current rectangle
                    #plt.imshow(gray.copy()[refPts[-2][1] : refPts[-1][1], refPts[-2][0] : refPts[-1][0]],
                    #           cmap='gray', interpolation='bicubic')
                    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                    #plt.show()
                    self.saveImage(gray.copy()[refPts[-2][1] : refPts[-1][1], refPts[-2][0] : refPts[-1][0]])
                    self.mouse.set_oldPts()
                    #self.mouse.reset()
                if intChar == 27:  # if esc key was pressed
                    # close all open windows
                    cv2.destroyAllWindows()
                    sys.exit()  # exit program



               
        
if __name__ == '__main__':

    ex = Example(sys.argv)
    ex.showDialog()

