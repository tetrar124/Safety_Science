    #! env python
# -*- coding: utf-8 -*-

import cv2
#from google.colab import drive
#drive.mount('/content/drive')
#%cd /content/drive/My\ Drive/colab/allPicGray
import os
import sys
import pylab as plt
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.utils import np_utils
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


class forToxicity(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')

    def main(self):
        os.chdir(r"G:\マイドライブ\colab")
        #% cd / content / drive / My\ Drive / colab /

        df = pd.read_csv('allDataAcute.csv')
        df = df[df['毒性値'] >0 ]
        print(df.head())
        dfArgae = df[(df['暴露時間（日）'] <= 4) & (df['栄養段階'] == '藻類')]
        dfArgae2 = dfArgae.groupby('CAS').median()
        dfFish = df[(df['暴露時間（日）'] <= 4) & (df['栄養段階'] == '魚類')]
        dfFish2 = dfFish.groupby('CAS').median()
        dfDaphnia = df[(df['暴露時間（日）'] <= 2) & (df['栄養段階'] == 'ミジンコ類')]
        dfDaphnia2 = dfDaphnia.groupby('CAS').median()
        roundList = []
        for i in dfFish2['毒性値']:
            if i < 1:
                roundList.append(0)
            elif 10 > i >= 1:
                roundList.append(1)
            elif 100 > i >= 10:
                roundList.append(2)
            elif 1000 > i >= 100:
                roundList.append(3)
            elif i >= 1000:
                roundList.append(4)
            else:
                print(i)
        print(roundList)
        print(len(roundList))
        dfFish2['toxicityGroup'] = roundList
        # dfFish2 =dfFish2.reset_index()
        print(dfFish2.head())
        plt.hist(roundList)
        plt.show()
        from keras.preprocessing.image import load_img, img_to_array
        x = []
        y = []
        i = 0
        eject1 = []
        for name in dfFish2.index:
            #fileName = '/content/drive/My Drive/colab/allPicGray/' + name + '.png'
            fileName = r"G:\マイドライブ\colab\allPicGray\\" +name + '.png'
            #if i < 10:
                try:
                    img = load_img(fileName, grayscale=True)
                    array = img_to_array(img).astype('float32') / 255
                    # print(array.shape)
                    # print(array)
                    x.append(array)
                    y.append(dfFish2['toxicityGroup'][name])
                    i += 1
                except:
                    eject1.append(name)
                    print(dfFish2['toxicityGroup'][name])
                    #print(array)
        print(eject1)
        print(y)
        import pickle
        with open('picX.txt', 'wb') as f:
            pickle.dump(x, f)
        with open('picY.txt', 'wb') as f:
            pickle.dump(y, f)
        # a = cv2.imread(fileName)
        # plt.imshow(a)
        # plt.show()
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,stratify=y)
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(x_train, y_train):
          print('TRAIN:', train_index, 'TEST:', test_index)

        def loadFile(self):
            import pickle
            with open('picY.txt', 'rb') as f:
                a = pickle.Unpickler(f)
                y = a.load()
            with open('picX.txt', 'rb') as f:
                a = pickle.Unpickler(f)
                x = a.load()
            return x,y
        def fileChack(self):
            import glob
            os.chdir(r'G:\マイドライブ\colab\allPicGray')
            files = glob.glob('*')
            names = []
            for name in files:
                tempName=name.replace('.png','')
                names.append(tempName)


if __name__ == '__main__':
    forToxicity=forToxicity()
    forToxicity.main()