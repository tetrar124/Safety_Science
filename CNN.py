#! env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import pylab as plt
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg

tf.enable_eager_execution()


class CNN(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')

    def makeCAStoPictures(self,df=None,strcture=None):
        os.chdir(r'G:\マイドライブ\Data\tox_predict\all_data')
        if strcture is None:
            strctureDf =pd.read_csv('G:\\マイドライブ\Data\\tox_predict\\all_data\\structure_result.csv',engine='python')
            df = strctureDf[['CAS','canonical_smiles']][strctureDf['CAS'].isin(strctureDf['CAS'].tolist())]
            #df = pd.merge(df,extAllDataDf)
        try:
            os.makedirs('allPictures')
        except:
            pass
        extract = zip(df['CAS'],df['canonical_smiles'])
        CAS =df['CAS'][0],
        smiles =df['canonical_smiles'][0]
        for CAS, smiles in extract:
            try:
                m = Chem.MolFromSmiles(smiles)
                #AllChem.Compute2DCoords(m)
                name = '.\\allPictures\\' + str(CAS) + '.png'
                #Draw.MolToFile(m, name)
                rdDepictor.Compute2DCoords(m)
                mc = Chem.Mol(m.ToBinary())
                Chem.Kekulize(mc)
                #mh = Chem.AddHs(m)
                drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
                drawer.SetFontSize(0.8)
                drawer.DrawMolecule(mc)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                SVG(svg.replace('svg:', ''))
                fw = open("out.svg", "w")
                fw.write(svg)
                fw.close()
                cairosvg.svg2png(url='out.svg', write_to=name)
            except:
                #     print("pass1")
                pass
    def forDataSetPicture(self):
        os.chdir(r'G:\マイドライブ\Data\tox_predict\all_data\allPictures')
        cas = [p.replace('.png', '') for p in glob.glob('*.png')]
        files = [os.path.abspath(p) for p in glob.glob('*.png')]
        dfFiles = pd.DataFrame({'CAS': cas, 'Files': files},index=False)
        #Acute Toxicity
        df =pd.read_csv(r'G:\マイドライブ\Data\tox_predict\all_data\allDataAcute.csv',engine= 'python',encoding='utf-8')
        targetToxValues = df[['CAS','毒性値','栄養段階']]
        toxitys = targetToxValues['毒性値'].tolist()
        toxitys1 = [1 if p < 0.01 else 0 for p in toxitys]
        toxitys2 = [2 if  1 > p >= 0.1 else 0 for p in toxitys]
        toxitys3 = [3 if  10 > p >= 1 else 0 for p in toxitys]
        toxitys4 = [4 if  100 > p >= 10 else 0 for p in toxitys]
        toxitys5 = [5 if  1000 > p >= 100 else 0 for p in toxitys]
        toxitys6 = [6 if  10000 > p >= 1000 else 0 for p in toxitys]
        toxitys7 = [7 if  p >= 10000 else 0 for p in toxitys]

        toxConnect = [a+b+c+d+e+f+g for a,b,c,d,e,f,g in zip(toxitys1,toxitys2,toxitys3,toxitys4,toxitys5,toxitys6,toxitys7)]
        df['label']= toxConnect
        df2 = pd.merge(df, dfFiles, on=['CAS'])
        df2.to_csv('forCNN.csv',index=False)

    def read(filename):
        img = tf.read_file(filename)
        return tf.image.decode_png(img, channels=3)
    filename=r'G:\マイドライブ\Data\tox_predict\all_data\allPictures\100-00-5.png'

    def write_jpg(data):
        sample = tf.image.encode_jpeg(data, format='rgb', quality=100)
        sample.numpy()
        filepath = "000.png"
        with open(filepath, 'wb') as fd:
            fd.write(sample)
    def rgb2Gray(self):
        os.chdir(r"G:\マイドライブ\Data\tox_predict\all_data\allPicturesSvg")
        
        import cv2
        import glob
        try:
            os.mkdir('allPicGray')
        for name in glob.glob('*.png'):
            temp = cv2.imread(name,0)
            name2  = '.\\allPicGray\\' +name
            cv2.imwrite(name2,temp)


    # sess = tf.Session()
    # init = tf.initialize_all_variables()
    # sess.run(tf.global_variables_initializer())
    # tf.train.start_queue_runners(sess)
    # x = sess.run(output)

    def main(self):
        os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
        return


if __name__ == '__main__':
    cnn=CNN()
    cnn.makeCAStoPictures()
