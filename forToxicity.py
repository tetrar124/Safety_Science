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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.utils import np_utils
import random

from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator

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
        dfFish2['toxicityGroup'] = roundList

        print(roundList)
        print(len(roundList))
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
                    #x.append(img)
                    y.append(dfFish2['toxicityGroup'][name])
                    i += 1
                except:
                    eject1.append(name)
                    print(dfFish2['toxicityGroup'][name])
                    #print(array)
        x = np.array(x,dtype='float')
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
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,stratify=y)
        from keras.utils import to_categorical
        y_test = to_categorical(y_test)
        y_train = to_categorical(y_train)

        kf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(x_train, y_train):
          print('TRAIN:', train_index, 'TEST:', test_index)

    train_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
    )
    train_generator.fit(x_train)

     class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(32,(3, 3), padding='same', activation=tf.nn.relu, data_format="channels_first")
            self.conv2 =  tf.keras.layers.Conv2D(32,(3, 3), padding='valid', activation=tf.nn.relu)
            self.mp1 =  tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            self.dropout1 = tf.keras.layers.Dropout(0.25)

            self.conv3 = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu)
            self.conv4 =  tf.keras.layers.Conv2D(64,(3, 3),activation=tf.nn.relu)
            self.mp2  =   tf.keras.layers.MaxPooling2D( pool_size=(2, 2))
            self.dropout2 = tf.keras.layers.Dropout(0.25)

            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense( 256, activation=tf.nn.relu)
            self.dropout3 = tf.keras.layers.Dropout(0.25)
            self.dense2 = tf.keras.layers.Dense(5,activation=tf.nn.softmax)
        def call(self, inputs,training=False):
            #x = tf.keras.layers.Input(shape=(200,200,1))
            x = self.conv1(inputs)
            x = self.conv2(x)
            x= self.mp1(x)
            if training: x = self.dropout1(x)
            x= self.dropout1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.mp2(x)
            if training: x = self.dropout2(x)
            x = self.dropout2(x)
            x = self.flatten(x)
            x = self.dense1(x)
            if training: x = self.dropout3(x)
            x = self.dropout3(x)
            return self.dense2(x)
    model = MyModel()
    num_classes=5
    model = tf.keras.Sequential([
        #tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, data_format="channels_first"),
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(200, 200,1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        # tf.keras.layers.Conv2D(128, (8, 8), activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(128, (8, 8), activation=tf.nn.relu),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
    model.summary()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    from tensorflow.python.keras.callbacks import TensorBoard
    #tsb = TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=True, write_images=True)
    summarizer=dict(
        directory='./logs-tensorboard',
        steps=50,
        labels=['graph', 'losses','batch_loss','total-loss']
    )
    #tensorboard --logdir=G:\\マイドライブ\\colab\\logs
    model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(learning_rate=0.001),metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=96, epochs=30, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./log/')],verbose=1,validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50)

    !python tensorflow/tensorboard/tensorboard.py --logdir=./logs


    def loadFile(self):
            os.chdir(r'G:\マイドライブ\colab')
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