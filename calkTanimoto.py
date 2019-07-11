from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import Draw
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
import pandas as pd
import numpy as np
import math
import pubchempy as pcp
import os
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


def calc_tanimoto(multi = None):
    #df = pd.read_csv('connect_result.csv')
    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
    df = pd.read_csv('extChronicStrcture.csv')
    result_df = pd.DataFrame(index=df['CAS'])
    #fin1 = df['cactvs_fingerprint']
    fin1 =df['isomeric_smiles'].fillna('')

    for i,fin_temp1_ori in enumerate(fin1):
        print(fin_temp1,i)
        col=[]
        if fin_temp1 =='':
            pass
        else:
            fin_temp1 =Chem.MolFromSmiles(fin_temp1_ori)
            if fin_temp1 == None:
                pass
            else:
                fin_temp1 = AllChem.GetMACCSKeysFingerprint(fin_temp1)
            trigger = 0
            for j,fin_temp2_ori in enumerate(fin1):
                # if multi == None and i >= j:
                #         result ='0'
                #         col.append(result)
                if trigger == 0 and fin_temp1_ori ==  fin_temp2_ori:
                    result =0
                    col.append(result)

                elif fin_temp1 == None or fin_temp2_ori =='':
                    result = '0'
                    col.append(result)
                else:
                    try:
                        fin_temp2 = Chem.MolFromSmiles(fin_temp2_ori)
                        fin_temp2= AllChem.GetMACCSKeysFingerprint(fin_temp2)
                        result = DataStructs.FingerprintSimilarity(fin_temp1,fin_temp2)
                        #print(result)
                        col.append(result)
                    except:
                        result = '0'
                        col.append(result)
        try:
            result_col_df = pd.DataFrame({df['CAS'][i]:col},index=df['CAS'])
            result_df[df['CAS'][i]] = result_col_df
        except:
            result_col_df = pd.DataFrame({df['CAS'][i]:''},index=df['CAS'])
            result_df[df['CAS'][i]] = result_col_df
        #print(result_df.head())
    #result_df.to_csv('MACCSKeys_tanimoto.csv')
    result_df.to_csv('chronicMACCSKeys_tanimoto.csv')

def countFiles():
    import os
    import os.path
    import glob
    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\DBSCAN-notox\\'
    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\GMM\\'
    os.chdir(path)
    folders = glob.glob('*')
    print(len(folders))
    count = []
    for dir in folders:
        path1 = path + dir
        os.chdir(path1)
        files = glob.glob('*')
        print(dir,len(files))
        count.append(len(files))
        os.chdir(path)
    count.sort()
    print(count)

def calc_tanimoto2(df,multi=None):
    import os

    #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
    #df = pd.read_csv('structure_result2.csv')
    result_df = pd.DataFrame(index=df['CAS'])

    #fin1 = df['cactvs_fingerprint']
    fin1 =df['isomeric_smiles'].fillna('')

    for i,fin_temp1_ori in enumerate(fin1):
        print(i)
        col=[]
        if fin_temp1_ori =='':
            pass
        else:
            molFile =Chem.MolFromSmiles(fin_temp1_ori)
            if molFile == None:
                pass
            else:
                finger1 = AllChem.GetMACCSKeysFingerprint(molFile)
            trigger = 0
            for j,fin_temp2_ori in enumerate(fin1):
                if trigger == 0 and fin_temp1_ori ==  fin_temp2_ori:
                    result =0
                    col.append(result)

                elif molFile == None or fin_temp2_ori =='':
                    result = '0'
                    col.append(result)
                else:
                    try:
                        molFile2 = Chem.MolFromSmiles(fin_temp2_ori)
                        finger2= AllChem.GetMACCSKeysFingerprint(molFile2)
                        result = DataStructs.FingerprintSimilarity(finger1,finger2)
                        #print(result)
                        col.append(result)
                    except:
                        result = '0'
                        col.append(result)
        try:
            result_col_df = pd.DataFrame({df['CAS'][i]:col},index=df['CAS'])
            #result_col_df = pd.DataFrame(col,colunms =df['CAS'][i], index=df['CAS'])

            result_df[df['CAS'][i]] = result_col_df
        except:
            result_col_df = pd.DataFrame({df['CAS'][i]:''},index=df['CAS'])
            result_df[df['CAS'][i]] = result_col_df
        #print(result_df.head())
    #result_df.to_csv('MACCSKeys_tanimoto.csv')
    result_df.to_csv('cembleChronicMACCSKeys_tanimoto.csv')
def calc_tanimoto_parallel(df,multi=None):
    #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
    #df = pd.read_csv('structure_result2.csv')
    fin1 =df['isomeric_smiles'].fillna('').tolist()

    def calcTani(i,fin_temp1_ori):
        print(i)
        col=[]
        if fin_temp1_ori =='':
            pass
        else:
            molFile =Chem.MolFromSmiles(fin_temp1_ori)
            if molFile == None:
                pass
            else:
                finger1 = AllChem.GetMACCSKeysFingerprint(molFile)
            trigger = 0
            for j,fin_temp2_ori in enumerate(fin1):
                if trigger == 0 and fin_temp1_ori ==  fin_temp2_ori:
                    result =0
                    col.append(result)

                elif molFile == None or fin_temp2_ori =='':
                    result = '0'
                    col.append(result)
                else:
                    try:
                        molFile2 = Chem.MolFromSmiles(fin_temp2_ori)
                        finger2= AllChem.GetMACCSKeysFingerprint(molFile2)
                        result = DataStructs.FingerprintSimilarity(finger1,finger2)
                        #print(result)
                        col.append(result)
                    except:
                        result = '0'
                        col.append(result)
            return  i,col

    r =Parallel(n_jobs=6)([delayed(calcTani)(i, smiles) for i, smiles in enumerate(tqdm(fin1))])
    r.sort(key=lambda x: x[0])
    tanimotoData = [t[1] for t in r]
    result_df = pd.DataFrame(tanimotoData, index=df['CAS'], columns=df['CAS'])
    result_df.to_csv('cembleChronicMACCSKeys_tanimoto.csv')

if __name__ == '__main__':
    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
    testDf = pd.read_csv(r'G:\マイドライブ\Data\Meram Chronic Data\paraTest.csv',engine='python')

    df1 = pd.read_csv(r'G:\マイドライブ\Data\Meram Chronic Data\extChronicStrcture.csv',engine='python')
    df1 = df1[['CAS','isomeric_smiles']]
    df1 = df1.dropna(how='any')

    df2 = pd.read_csv(r'G:\マイドライブ\Data\tox_predict\chemble\extractSmiles.csv',engine='python',header=None)
    df2.columns =  ['CAS','isomeric_smiles']
    df2['CAS'] = df2['CAS'].map(lambda x : str(x) + '-dummy')
    num = np.random.randint(0,100000,10000)
    df2 = df2.iloc[num,:]
    df = pd.concat([df1,df2])
    df.reset_index(drop=True, inplace=True)
    df['CAS'].astype(str)
    calc_tanimoto_parallel(df)
    #countFiles()