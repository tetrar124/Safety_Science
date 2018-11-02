#! env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import pylab as plt
import numpy as np
# Safety_Science.pic_calc
# Date: 2018/07/21
# Filename: pic_calc 

__author__ = 'tatab'
__date__ = "2018/07/21"


class pic_calc(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')

    def main(self):
        path =r"G:\マイドライブ\TEM\2017年NEDO\20180608CNF\日本製紙\塩酸入りカーボンルテ5ul(計測用)"
        os.chdir(path)
        name = 'Tv28.jpg'
        # plt.subplot(1,3,1)
        im = cv2.imread(name)
        # plt.imshow(im)
        plt.axis('off')
        plt.subplot(3,2,1)
        def openAndClose(im):
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            return closing
        imGray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        plt.imshow(imGray,'gray')
        plt.title('imgray')

        #copy
        imGray2 = imGray
        imGray3 = imGray

        '''cut test'''
        plt.subplot(3,2,2)
        imGrayCut = (imGray>20) & (imGray<150)
        print(imGrayCut)
        imGray3[imGrayCut ] = 0
        #imGray3 = openAndClose(imGray3)
        imGray3 = cv2.GaussianBlur(imGray3, (3, 3), 0)
        cv2.imwrite("test.png", imGray3)
        plt.imshow(imGray3,'gray')
        plt.title('cut')
        plt.axis('off')



        plt.subplot(3,2,3)
        line = (imGray > 2) & (imGray <= 170)
        plt.imshow(line,'gray')
        print(imGray.shape)
        plt.axis('off')
        #
        plt.subplot(3,2,4)
        plt.hist(imGray.flat, bins=255, range=(0, 255))
        ret, th1 = cv2.threshold(imGray,2, 200, cv2.THRESH_BINARY)
        plt.subplot(3,2,5)
        plt.imshow(th1,'gray')
        num,contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(imGray.shape,np.uint8)
        plt.subplot(3, 2, 6)
        result = cv2.drawContours(imGray2,contours,-1, 255, -1)
        plt.imshow(result,'gray')

        #エッジ検出
        # plt.subplot(3,2,6)
        # imgGau = cv2.GaussianBlur(imGray3, (3, 3), 0)
        # lap = cv2.Laplacian(imgGau, cv2.CV_32F)
        # edge_lap = cv2.convertScaleAbs(lap)
        # plt.imshow(edge_lap)

        plt.show()

if __name__ == '__main__':
    pic_calc =pic_calc()
    pic_calc.main()