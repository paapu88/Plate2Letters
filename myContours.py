

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
        self.kill_index = set()      # index list of contours to be removed

    def setGray(self, image_in=None):
        if image_in is None:
            clone = self.image.copy()
        else:
            clone = image_in.copy()
        try:
            self.gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        except:
            # in case it was gray already
            self.gray = clone

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
            if self.blur is not None:
                clone = self.getBlur()
            else:
                clone = self.getGray()
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

        #cv2.imshow("imgThresh", self.thres.copy())
        #intChar = cv2.waitKey(0)

    def getThreshold(self):
        if self.thres is not None:
            return self.thres.copy()
        elif self.blur is not None:
            return self.blur.copy()
        elif self.gray is not None:
            return self.gray.copy()
        else:
            return self.image.copy()

    def setContours(self, image_in=None):
        if image_in is None:
            clone = self.getThreshold()
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

    def setContoursOnly(self, npaContours=None):
        self.npaContours = npaContours

    def getLenContours(self):
        return len(self.npaContours)

    def getKillIndex(self):
        return self.kill_index

    def checkContourHeight(self, accept_ratio=0.3):
        """ mark too small in height to be killed """
        for i, (npaContour, npaHierarchy) in \
            enumerate(zip(self.npaContours, self.npaHierarchies)):
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(npaContour)
            if intHeight/self.image.shape[0] < accept_ratio:
                self.kill_index.add(i)

    def checkContourRatio(self, max_ratio=None, min_ratio=None):
        """" check whether the y/x ratio of the contour is bad """
        for i, (npaContour, npaHierarchy) in \
            enumerate(zip(self.npaContours, self.npaHierarchies)):
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(npaContour)
            if ((intHeight / intWidth ) > max_ratio) or \
               ((intHeight / intWidth ) < min_ratio):
                #print("ratio:", str(intHeight / intWidth ))
                self.kill_index.add(i)

    def checkContourArea(self, min_area=1, max_area=100000):
        """" check whether the are of contour is too small """
        # for each contour
        for i, (npaContour, npaHierarchy) in \
            enumerate(zip(self.npaContours, self.npaHierarchies)):
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(npaContour)
            print("A",intX, intY, intWidth, intHeight,min_area, max_area)
            if (intWidth * intHeight) < min_area:
                self.kill_index.add(i)
            elif (intWidth * intHeight) > max_area:
                self.kill_index.add(i)


    def get_standard_deviation(self, index, mykey='x', divide='y'):
        """ NOT USED: calculate standard deviation of contours """
        if mykey=='x':
            feature=0
        else:
            raise NotImplementedError("NOT IMPLEMENTED in getStandardDeviation")
        vec = np.zeros(len(index))
        heights = np.zeros(len(index))
        for i, ind in enumerate(index):
            vec[i] = cv2.boundingRect(self.npaContours[i])[2]
            heights[i] = cv2.boundingRect(self.npaContours[i])[3]
        result = np.std(vec)
        if divide=='y':
            result = result/np.mean(heights)
        return result

    def checkSubsequent(self, mykey='x', mymin=0.5, mymax=2.5, ratio=1.5):
        """Check that contours are near enought each other,
        use hight of a letter, because that is more constant"""
        ok = True
        if mykey == 'x':
            x_previous = cv2.boundingRect(self.npaContours[0])[0]
            for contour in self.npaContours[1:]:

                x_next = cv2.boundingRect(contour)[0]
                diff = x_next - x_previous
                bredth = cv2.boundingRect(contour)[3]/ratio
                #print("dx check:", diff, bredth)
                if (diff > mymax * bredth) or (diff < mymin * bredth):
                    ok = False
                    break
                x_previous = x_next
        else:
            raise NotImplementedError("only x implemented in chekckSubsequent")
        return ok


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
                self.kill_index.add(i)

    def markNonSetContours(self, myset=None):
        """ contours with indexes NOT belonging to the set are marked
            for removal """
        for i, npaContour in enumerate(self.npaContours):        
            if i not in list(myset):
                self.kill_index.add(i)

    def removeMarkedContours(self):
        """ remove contours that have been listed in kill_index 
            also remove corresponding hierarchy indexes"""

        self.npaContours = np.delete(
            self.npaContours, list(self.kill_index), axis=0)
        self.npaHierarchies = np.delete(
            self.npaHierarchies, list(self.kill_index), axis=0)
        self.kill_index = set()

    def sortContours(self, mykey='x'):
        """ sort contours by some position criterium """

        if mykey=='x': 
            feature=0
        else:
            raise NotImplementedError("NOT IMPLEMENTED in sortContours")

        #print("SHAPE:",self.npaContours.shape)
        #print(self.npaContours[0])
        mykeys = []        
        for npaContour in self.npaContours:
            mykeys.append(cv2.boundingRect(npaContour)[feature])
        np_mykeys=np.array(mykeys)
        #print("np_mykeys: ",np_mykeys)
        #print("shapes", self.npaContours.shape, np_mykeys.shape)
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
            raise NotImplementedError("NOT IMPLEMENTED in defineSets")

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

    def getInterception(self, set1=None, set2=None):
        """get interception of two sets, arrangel resulting sets so that longest comes first"""

        ok_sets = []
        for s1 in set1:
            for s2 in set2:
                myset = s1.intersection(s2)
                if len(myset)==6 and myset not in ok_sets:
                    ok_sets.append(myset)
                #lets take only 6-plates to start with
                #elif len(myset)==5 and myset not in ok_sets:
                #    ok_sets.append(myset)

        ok_lists=[]
        for ok_set in ok_sets:
            ok_lists.append(list(ok_set))
        #print("1",ok_lists)
        #print("2a",len(ok_lists), ok_lists.sort(key=len))
        return sorted(ok_lists, key=len, reverse=True)
