import numpy as np
import pandas as pd
import os
import glob
class Connect():
    def TempConnect(self):
        os.chdir('G:\\マイドライブ\\export\\01pibot-csv')
        df1 = pd.read_csv('resultグリコーゲン0.1%50ug__25_1col.csv',engine='python')
        df2 = pd.read_csv('result日本製紙0.1%50ug__5_1col.csv',engine='python')
        df1.columns= ['time','mass','glycogen']
        df2.columns= ['time','mass','Tempo']

        result = pd.merge(df1,df2,on=['time','mass'],how='outer')
        result['calc']= result['Tempo']-result['glycogen']

        result.to_csv('glycogenVStempo.csv',index=False)
    def eraseBase(self):
        os.chdir(r"C:\OneDrive\共有\解析結果")
        def pivot1():
            basefile = glob.glob(r"base\*pivot*")
            targetfiles = glob.glob(r"target\*pivot*")
            baseDf = pd.read_csv(basefile[0],index_col='time',engine='python')
            for target in targetfiles:
                beforeDf = pd.read_csv(target,index_col='time',engine= 'python')
                calcDf = beforeDf - baseDf
                calcDf[ calcDf < 0 ] = 0
                fileName = target.replace('.csv','') + '_delBase' + '.csv'
                calcDf.to_csv(fileName)
                print(fileName)
        def colOne():
            basefile = glob.glob(r"base\*1col*")
            targetfiles = glob.glob(r"target\*1col*")
            baseDf = pd.read_csv(basefile[0],engine='python')
            baseDf.columns = ['time','mass',basefile[0].replace('.csv','')]
            for target in targetfiles:
                beforeDf = pd.read_csv(target,engine= 'python')
                beforeDf.columns = ['time','mass',target.replace('.csv','')]
                baseDf = pd.merge(baseDf, beforeDf, on=['time', 'mass'], how='outer')
            baseDf.to_csv('target\\allConnect.csv')
        result_1col = colOne()
        #result_pivot = pivot1()

if __name__ == '__main__':
    tempconnect= Connect()
    tempconnect.eraseBase()