

import cv2
import numpy as np

class ContoursWithFilters():
    """ contours by opencv. They can be selected by criteria. 
        The figure they are based can be modified """

    
    def __init__(self, image=None, blurSize=(5,5),
                 thres_pixel_neib=5, thres_pixel_sub=2):
        self.image = image           # original image
        self.gray  = None            # image, gray
        self.blurSize=blurSize       # pixels are blurred to this
        self.blur = None             # image, blurred
        self.thres = None            # image, thersholded
        # Threshold: size of a pixel neighborhood for threshold value
        self.thres_pixel_neib = thres_pixel_neib
        # Threshold: constant subtracted from the mean or weighted mean
        self.thres_pixel_sub = thres_pixel_sub
        self.npaContours = None      # contours
        self.npaHierarchies = None   # hierarcly of contours
        self.kill_index = []         # index list of contours to be removed

    def setGray(self, image_in=None):
        if image_in is None:
            clone = self.image.copy()
        else:
            clone = image_in.copy()
        self.gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    def getImage(self):
        return self.image.copy()
        
    def getGray(self):
        return self.gray.copy()

    def setBlur(self, image_in=None, blurSize=None):
        if image_in is None:
            clone = self.gray.copy()
        else:
            clone = image_in.copy()
        if blurSize is not None:
            self.blurSize = blurSize
        self.blur = cv2.GaussianBlur(clone, self.blurSize, 0)

    def getBlur(self):
        return self.blur.copy()


    def setThreshold(self, image_in=None,
                     thres_pixel_neib=None, thres_pixel_sub=None):
        if image_in is None:
            clone = self.blur.copy()
        else:
            clone = image_in.copy()
        if thres_pixel_neib is not None:
            self.thres_pixel_neib = thres_pixel_neib
        if thres_pixel_sub is not None:
            self.thres_pixel_sub = thres_pixel_sub
        #dummy ,self.thres \
        #    = cv2.threshold(clone,110,255,cv2.THRESH_BINARY_INV)            

        self.thres = cv2.adaptiveThreshold(clone,    # input image
                                          # make pixels that pass
                                          # the threshold full white
                                          255,      
                                          # use gaussian rather than mean
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          # invert so foreground will be white
                                          cv2.THRESH_BINARY_INV,
                                          # size of a pixel neighborhood used
                                          # to calculate threshold value
                                          self.thres_pixel_neib,
                                          # constant subtracted from the mean
                                          # or weighted mean
                                          self.thres_pixel_sub)

        cv2.imshow("imgThresh", self.thres.copy())
        intChar = cv2.waitKey(0) 

    def getThreshold(self):
        return self.thres.copy()                
        
    def setContours(self, image_in=None):
        if image_in is None:
            clone = self.thres.copy()
        else:
            clone = image_in.copy()
        dummy, self.npaContours, self.npaHierarchies\
            = cv2.findContours(clone,
                               # retrieve the hierarchy of contours
                               cv2.RETR_TREE,
                               # leave only end points
                               cv2.CHAIN_APPROX_SIMPLE)
        #remove unnecessary dummy dimension
        self.npaHierarchies = np.squeeze(self.npaHierarchies)

    def getContours(self):
        return self.npaContours, self.npaHierarchies

    def getContoursOnly(self):
        return self.npaContours

    def getLenContours(self):
        return len(self.npaContours)

    def getKillIndex(self):
        return self.kill_index
        
    def checkContourRatio(self, max_ratio=None, min_ratio=None,
                          boundingRect=None):
        """" check whether the y/x ratio of the contour is bad """
        [intX, intY, intWidth, intHeight] = boundingRect        
        if ((intHeight / intWidth ) > max_ratio) or \
           ((intHeight / intWidth ) < min_ratio):
            #print("ratio:", str(intHeight / intWidth ))
            return True
        else:
            return False

    def checkContourArea(self, min_area=None, boundingRect=None):
        """" check whether the are of contour is too small """
        [intX, intY, intWidth, intHeight] = boundingRect        
        if (intWidth * intHeight) < min_area:
            #print("area", intWidth * intHeight, min_area)
            return True
        else:
            return False
    

    def markTooSmallorWrongAspectRatio(self,
                                       min_area=100,
                                       min_ratio=1.3,
                                       max_ratio=8.0):
        """ mark contours that are too small or have wrong aspect ratio """

        # for each contour
        for i, (npaContour, npaHierarchy) in \
            enumerate(zip(self.npaContours, self.npaHierarchies)):
            boundingRect=cv2.boundingRect(npaContour)         
            if self.checkContourRatio(max_ratio=max_ratio,
                                      min_ratio=min_ratio,
                                      boundingRect=boundingRect) \
                or self.checkContourArea(min_area=min_area, 
                                         boundingRect=boundingRect):
                self.kill_index.append(i)

                


    def setOrphans(self):
        """ set contours whose parent has been killed as orphans """
        # for each contour
        for npaHierarchy in self.npaHierarchies:
            my_parent = npaHierarchy[-1]
            if my_parent in self.kill_index:
                npaHierarchy[-1] = -1

    def markContoursWithParents(self):
        """ contours who have parents are marked for removal """
        for i, npaHierarchy in enumerate(self.npaHierarchies):        
            my_parent = npaHierarchy[-1]
            if my_parent != -1:
                self.kill_index.append(i)

    def markNonSetContours(self, myset=None):
        """ contours with indexes NOT belongint to the set are marked
            for removal """
        for i, npaContour in enumerate(self.npaContours):        
            if i not in myset:
                self.kill_index.append(i)                

    def removeMarkedContours(self):
        """ remove contours that have been listed in kill_index 
            also remove corresponding hierarchy indexes"""
        self.npaContours = np.delete(
            self.npaContours, self.kill_index, axis=0)
        self.npaHierarchies = np.delete(
            self.npaHierarchies, self.kill_index, axis=0)
        self.kill_index = []

    def sortContours(self, mykey='x'):
        """ sort contours by some position criterium """

        if mykey=='x': 
            feature=0
        else:
            raise notImplementedError("NOT IMPLEMENTED in sortContours")

        print("SHAPE:",self.npaContours.shape)
        #print(self.npaContours[0])
        mykeys = []        
        for npaContour in self.npaContours:
            mykeys.append(cv2.boundingRect(npaContour)[feature])
        np_mykeys=np.array(mykeys)
        print("np_mykeys: ",np_mykeys)
        print("shapes", self.npaContours.shape, np_mykeys.shape)
        sorted_idx = np.argsort(np_mykeys)
        self.npaContours = self.npaContours[sorted_idx]


    def defineSets(self, criterium='ypos', tolerance=0.3):
        """ group contour indexes to list of sets based on given criterium """

        sets=[]
        if criterium == 'ypos':
            feature = 1
        elif criterium == 'height':
            feature = 3
        else:
            raise notImplementedError("NOT IMPLEMENTED in defineSets")

        for i, npaContour in enumerate(self.npaContours):
            f1= cv2.boundingRect(npaContour)[feature]
            myset = set()
            for j, npaContour2 in enumerate(self.npaContours):
                f2= cv2.boundingRect(npaContour2)[feature]                   
                if (1-tolerance)*f1 < f2 and \
                   (1+tolerance)*f1 > f2:
                    myset.add(j)
                #print(i,j,f1,f2)

            if myset not in sets:
                sets.append(myset)
        #print("defined SETS", self.npaContours.shape)
        return sets