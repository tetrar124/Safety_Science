import os
import sys
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit.Chem import Draw
from rdkit import rdBase
import numpy as np
import statistics

class testCluster(object):
    def sepTestData(self,df=None,):
        if df is None:
            data = [[1, 2, 3],
                [50, 60, 70],
                [1,2,3],
                [1,2,3],
                [2,4,6],
                [3,4,3],
                [2,4,6],
                [1,2,3],
                [1,3,4],
                [1,2,3]]
            df = pd.DataFrame(data)
            df.index = np.arange(0,df.shape[0],1)
            print(df)
        else:
            pass
        print(df.shape)
        df2 = df.copy()
        df2['sum'] = df.sum(axis=1)
        df2 = df2[df2['sum'] != 0 ]
        df2 = df2.drop('sum',axis = 1)
        dfAllData = pd.read_csv('allData.csv',engine='python',encoding='utf-8')
        dfAllData = dfAllData.drop(dfAllData.index[dfAllData['毒性値'] < 0])
        dfAllDataCount = dfAllData['CAS'].value_counts()
        dfAllDataCount = dfAllDataCount[dfAllDataCount>= 20]
        print(len(dfAllDataCount))
        #サンプリング
        testDataCAS = dfAllDataCount.sample(n = 21 , random_state=7)
        testData = df2[df2['CAS'].isin(testDataCAS.index.tolist())]
        #残りデータ
        clusterData = pd.concat([df,testData]).drop_duplicates(keep=False)
        clusterData = clusterData.drop(clusterData.columns[list(testData.index.values+1)], axis=1)
        testData = testData.drop(testData.columns[list(testData.index.values+1)], axis=1)

        #testData.to_csv('testData.csv',header=False)
        testDataMeltDf = pd.melt(testData, id_vars=['CAS'])
        samples = testDataMeltDf['CAS'].unique()

        toxValues = pd.read_csv('allData.csv',engine='python',encoding="UTF-8")
        resultDf = pd.DataFrame(columns=['CAS','variable','value'])
        columnList = ['targetCAS','targetName','Fish_Count','Fish_Substances','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances', 'Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        sampleToxValues =pd.DataFrame(columns=columnList)
        for sample in samples:
            toxValueList = []
            top = testDataMeltDf[testDataMeltDf['CAS']==sample].sort_values('value',ascending=False).head()
            toxValuesTemp=toxValues[toxValues['CAS']==sample]
            toxValueList.extend([sample,toxValuesTemp ['化学物質名'].values[0]])
            for name in ['魚類','ミジンコ類','藻類']:
                targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階']==name]
                toxValueList.extend([len(targetToxValues['毒性値']), len(targetToxValues['化学物質名'].unique())])
                try:
                    targetToxValues['毒性値'].values[0]
                    toxValueList.extend([targetToxValues['毒性値'].median(),targetToxValues['毒性値'].min(),targetToxValues['毒性値'].max(),targetToxValues['毒性値'].mode()[0],targetToxValues['毒性値'].mean()])
                except:
                    toxValueList.extend(['nan', 'nan', 'nan', 'nan','nan'])
            print(toxValueList)
            tempDf = pd.DataFrame([toxValueList],columns=columnList)
            sampleToxValues = pd.concat([sampleToxValues,tempDf])

            resultDf = pd.concat([resultDf,top])
        resultDf.to_csv('result.csv',index=False)
        clusterData.to_csv('clusterData.csv',index=False)
        sampleToxValues.to_csv('targetToxValues.csv',index=False)

    def NewMergeCalcTox(self):
        # df1 = pd.read_csv('clusterData_Louvain_cluster_08.csv')
        # df2 = pd.read_csv('allData.csv')

        df1 = pd.read_csv('louvain_8071.0_150_th0.426.csv')
        df2 = pd.read_csv('extChronicData.csv',encoding='cp932')
        #df2 = df2[df2['毒性値']!='繁殖']
        #df2 = df2['毒性値'].astype(float)
        #df2 = df2.drop(df2.index[df2['毒性値']<0])
        df2 = df2[df2['毒性値']>=0]
        toxValues = pd.merge(df1,df2,on = 'CAS',how = 'outer')
        #toxValues.to_csv('allDataAndCluster.csv')
        '''クラスタ帰属物質のデータ抽出'''
        clusters = toxValues['cluster'].unique()
        columnList = ['cluster','Fish_Count','Fish_Substances','Fish_Peak','Fish_result','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_Peak','Daphnia_result','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances','Algae_Peak', 'Algae_result','Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        sampleToxValues =pd.DataFrame(columns=columnList)
        for cluster in clusters:
            if cluster == 'nan':
                pass
            else:
                toxValueList = []
                toxValuesTemp=toxValues[toxValues['cluster']==cluster]
                toxValueList.extend([cluster,])
                for name in ['魚類','ミジンコ類','藻類']:
                    targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階']==name]
                    toxValueList.extend([len(targetToxValues['毒性値']),len(targetToxValues['化学物質名'].unique())])
                    if name is '魚類':
                        targetToxValues = targetToxValues[targetToxValues['暴露時間（日）'] <= 4]
                    elif name is 'ミジンコ類':
                        targetToxValues = targetToxValues[targetToxValues['暴露時間（日）'] <= 2]
                    elif name is '藻類':
                        targetToxValues = targetToxValues[targetToxValues['暴露時間（日）'] <= 4]
                    try:
                        minList =list(filter(lambda x:  x <0.01  , targetToxValues['毒性値']))
                        PointZeroToOne =list(filter(lambda x:  0.1>x >=0.01  , targetToxValues['毒性値']))
                        ZeroPointToOne =list(filter(lambda x:  1>x >=0.1  , targetToxValues['毒性値']))
                        OneToTen = list(filter(lambda x: 10 > x >=1  , targetToxValues['毒性値']))
                        TenToHun = list(filter(lambda x: 100 > x >=10  , targetToxValues['毒性値']))
                        HunToThou = list(filter(lambda x: 1000 > x >= 100, targetToxValues['毒性値']))
                        ThouToTenThou = list(filter(lambda x: 10000 > x >= 1000, targetToxValues['毒性値']))
                        maxList = list(filter(lambda x: x >= 10000, targetToxValues['毒性値']))
                        result = []
                        peakCount = 0
                        for tox in [minList,PointZeroToOne,ZeroPointToOne,OneToTen,TenToHun,HunToThou,ThouToTenThou,maxList]:
                            if len(result) == 0:
                                result = tox
                                temp = tox
                                peakCount = 1
                            elif len(result) > len(tox):
                                pass
                            elif len(result) < len(tox):
                                result = tox
                                temp = tox
                                peakCount = 1
                            elif len(result) == len(tox):
                                result = tox
                                temp.extend(tox)
                                peakCount += 1
                        result = temp
                        try:
                            resultValue = statistics.median(result)
                        except:
                            resultValue = 'nan'
                        if peakCount ==1 :
                            peak = 'Single Top'
                        elif peakCount == 2:
                            peak = 'Double Top'
                        elif peakCount ==3:
                            peak ='Triple Top'
                        elif peakCount > 3:
                            peak = 'Flat'
                        #print(resultValue)
                        toxValueList.extend([peak,resultValue,targetToxValues['毒性値'].median(),targetToxValues['毒性値'].min(),targetToxValues['毒性値'].max(),targetToxValues['毒性値'].mode()[0],targetToxValues['毒性値'].mean()])
                    except:
                        if len(targetToxValues['毒性値']) == 1:
                            val = targetToxValues['毒性値']
                            toxValueList.extend(['-',val, val, val, val, val, val])
                        else:
                            toxValueList.extend(['-','nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
                #print(toxValueList)
                tempDf = pd.DataFrame([toxValueList],columns=columnList)
                sampleToxValues = pd.concat([sampleToxValues,tempDf])

        df3 = pd.read_csv('result.csv')
        df3.columns = ['targetCAS','CAS','TanimotoValue']
        #df4 = pd.read_csv('clusterData_Louvain_cluster_08.csv')
        df4 = pd.read_csv('louvain_8071.0_150_th0.426.csv')

        nearClusterDf = pd.merge(df3,df4,how='left',on='CAS')
        nearClusterDf = pd.merge(nearClusterDf,sampleToxValues,how='left',on='cluster')
        clusterNullDf = nearClusterDf[nearClusterDf['cluster'].isnull()]
        columnList = ['targetCAS','CAS','TanimotoValue','cluster','Fish_Count','Fish_Substances','Fish_Peak','Fish_result','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_Peak','Daphnia_result','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances','Algae_Peak', 'Algae_result', 'Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        casToxDf = pd.DataFrame(columns=columnList)

        '''クラスタなし集計'''
        for target,cas,value,cluster in clusterNullDf[['targetCAS','CAS','TanimotoValue','cluster']].values:
            print(cas)
            toxValueList = []
            toxValuesTemp = toxValues[toxValues['CAS'] == cas]
            toxValueList.extend([target,cas,value,cluster ])
            for name in ['魚類', 'ミジンコ類', '藻類']:
                targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階'] == name]
                toxValueList.extend([len(targetToxValues['毒性値']), len(targetToxValues['化学物質名'].unique())])
                # if name is '魚類' :
                #     targetToxValues = targetToxValues[targetToxValues['暴露時間（日）']<=4]
                # elif name is 'ミジンコ類':
                #     targetToxValues = targetToxValues[targetToxValues['暴露時間（日）']<=2]
                # elif name is '藻類':
                #     targetToxValues = targetToxValues[targetToxValues['暴露時間（日）']<=4]
                try:
                    minList = list(filter(lambda x: x < 0.01, targetToxValues['毒性値']))
                    PointZeroToOne = list(filter(lambda x: 0.1 > x >= 0.01, targetToxValues['毒性値']))
                    ZeroPointToOne = list(filter(lambda x: 1 > x >= 0.1, targetToxValues['毒性値']))
                    OneToTen = list(filter(lambda x: 10 > x >= 1, targetToxValues['毒性値']))
                    TenToHun = list(filter(lambda x: 100 > x >= 10, targetToxValues['毒性値']))
                    HunToThou = list(filter(lambda x: 1000 > x >= 100, targetToxValues['毒性値']))
                    ThouToTenThou = list(filter(lambda x: 10000 > x >= 1000, targetToxValues['毒性値']))
                    maxList = list(filter(lambda x: x >= 10000, targetToxValues['毒性値']))
                    result = []
                    peakCount = 0
                    for tox in [minList, PointZeroToOne, ZeroPointToOne, OneToTen, TenToHun, HunToThou, ThouToTenThou,
                                maxList]:
                        if len(result) == 0:
                            result = tox
                            temp = tox
                            peakCount = 1
                        elif len(result) > len(tox):
                            pass
                        elif len(result) < len(tox):
                            result = tox
                            temp = tox
                            peakCount = 1
                        elif len(result) == len(tox):
                            result = tox
                            temp.extend(tox)
                            peakCount += 1
                    result = temp
                    resultValue = statistics.mean(result)
                    if peakCount == 1:
                        peak = 'Single Top'
                    elif peakCount == 2:
                        peak = 'Double Top'
                    elif peakCount == 3:
                        peak = 'Triple Top'
                    elif peakCount > 3:
                        peak = 'Flat'
                    #print(resultValue)
                    toxValueList.extend(
                        [peak, resultValue, targetToxValues['毒性値'].median(), targetToxValues['毒性値'].min(),
                         targetToxValues['毒性値'].max(), targetToxValues['毒性値'].mode()[0], targetToxValues['毒性値'].mean()])
                except:
                    if len(targetToxValues['毒性値']) == 1:
                        val = targetToxValues['毒性値']
                        toxValueList.extend(['-', val, val, val, val, val, val])
                    else:
                        toxValueList.extend(['-', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
            print(toxValueList)
            tempCasDf = pd.DataFrame([toxValueList], columns=columnList)
            casToxDf = pd.concat([casToxDf, tempCasDf])
        casToxDf = pd.concat([casToxDf, tempCasDf])
        nearClusterDf = nearClusterDf[np.isfinite(nearClusterDf['cluster'])]
        nearClusterDf = pd.concat([nearClusterDf,casToxDf]).sort_values(by='targetCAS')
        nearClusterDf = nearClusterDf.drop_duplicates()
        nearClusterDf.to_csv('predict08_.csv',index = False)

    def mergeCalcTox(self):
        df1 = pd.read_csv('clusterData_Louvain_cluster_08.csv')
        df2 = pd.read_csv('allData.csv')
        df2 = df2.drop(df2.index[df2['毒性値']<0])
        toxValues = pd.merge(df1,df2,on = 'CAS',how = 'outer')
        #toxValues.to_csv('allDataAndCluster.csv')

        clusters = toxValues['cluster'].unique()
        columnList = ['cluster','Fish_Count','Fish_Substances','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances', 'Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        sampleToxValues =pd.DataFrame(columns=columnList)
        for cluster in clusters:
            if cluster == 'nan':
                pass
            else:
                toxValueList = []
                toxValuesTemp=toxValues[toxValues['cluster']==cluster]
                toxValueList.extend([cluster,])
                for name in ['魚類','ミジンコ類','藻類']:
                    targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階']==name]
                    toxValueList.extend([len(targetToxValues['毒性値']),len(targetToxValues['化学物質名'].unique())])
                    try:
                        targetToxValues['毒性値'].values[0]
                        toxValueList.extend([targetToxValues['毒性値'].median(),targetToxValues['毒性値'].min(),targetToxValues['毒性値'].max(),targetToxValues['毒性値'].mode()[0],targetToxValues['毒性値'].mean()])
                    except:
                        toxValueList.extend(['nan', 'nan', 'nan', 'nan','nan'])
                print(toxValueList)
                tempDf = pd.DataFrame([toxValueList],columns=columnList)
                sampleToxValues = pd.concat([sampleToxValues,tempDf])
        df3 = pd.read_csv('result.csv')
        df3.columns = ['targetCAS','CAS','TanimotoValue']
        df4 = pd.read_csv('clusterData_Louvain_cluster_08.csv')
        nearClusterDf = pd.merge(df3,df4,how='left',on='CAS')
        nearClusterDf = pd.merge(nearClusterDf,sampleToxValues,how='left',on='cluster')
        clusterNullDf = nearClusterDf[nearClusterDf['cluster'].isnull()]
        columnList = ['targetCAS','CAS','TanimotoValue','cluster','Fish_Count','Fish_Substances','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances', 'Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        casToxDf = pd.DataFrame(columns=columnList)
        for target,cas,value,cluster in clusterNullDf[['targetCAS','CAS','TanimotoValue','cluster']].values:
            toxValueList = []
            toxValuesTemp = toxValues[toxValues['CAS'] == cas]
            toxValueList.extend([target,cas,value,cluster ])
            for name in ['魚類', 'ミジンコ類', '藻類']:
                targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階'] == name]
                toxValueList.extend([len(targetToxValues['毒性値']), len(targetToxValues['化学物質名'].unique())])
                if name is '魚類' :
                    targetToxValues = targetToxValues[targetToxValues['暴露時間（日）']<5]
                else:
                    pass
                try:
                    targetToxValues['毒性値'].values[0]
                    toxValueList.extend(
                        [targetToxValues['毒性値'].median(), targetToxValues['毒性値'].min(), targetToxValues['毒性値'].max(),
                         targetToxValues['毒性値'].mode()[0], targetToxValues['毒性値'].mean()])
                except:
                    toxValueList.extend(['nan', 'nan', 'nan', 'nan', 'nan'])
            print(toxValueList)
            tempCasDf = pd.DataFrame([toxValueList], columns=columnList)
            casToxDf = pd.concat([casToxDf, tempCasDf])
        casToxDf = pd.concat([casToxDf, tempCasDf])
        nearClusterDf = pd.concat([nearClusterDf,clusterNullDf]).drop_duplicates(keep=False)
        nearClusterDf = pd.concat([nearClusterDf,casToxDf]).sort_values(by='targetCAS')
        nearClusterDf.to_csv('predict08_.csv',index = False)

    def connectNames(self):
        nearClusterDf = pd.read_csv('predict08_.csv',engine='python')
        nearClusterDf = nearClusterDf.rename(columns ={'CAS':'similarStructureCAS'})
        allData = pd.read_csv('extChronicData.csv',engine='python',encoding='cp932')
        #allData = pd.read_csv('allData.csv',engine='python',encoding='utf-8')
        casNumbers = nearClusterDf[['targetCAS','similarStructureCAS']].values.tolist()
        nameList = []
        for target,cas in casNumbers:
            targetName = allData['化学物質名'][allData['CAS']== target].tolist()[0]
            casName = allData['化学物質名'][allData['CAS']== cas].tolist()[0]
            nameList.extend([[target,cas,targetName,casName]])
            print(targetName,casName)
        nameDf = pd.DataFrame(nameList,columns=['targetCAS','similarStructureCAS','targetName','similarStructureName'])
        #nameDf.to_csv('nameDf.csv')
        nearClusterDf =nearClusterDf.rename(columns={'target':'targetCAS','CAS':'similarStructureCAS','value':'TanimotoValue'})
        print(nearClusterDf.columns)
        resultDf = pd.merge(nameDf,nearClusterDf,on=['targetCAS','similarStructureCAS'])
        print('resultDf',resultDf.columns)
        resultDf = resultDf.drop_duplicates()
        resultDf.to_csv('predict08withName.csv',index=False,encoding='utf-8')

    def checkDupe(self,df):
        #df2 = df[df[df==1].any(axis=1)]
        df = df.set_index('CAS')
        dfBool = df.where(df==1,0).fillna(0)
        maxCheck =[]
        columns = np.arange(0,17,1).tolist()
        resultDf = pd.DataFrame(columns=columns)
        for i in np.arange(0,dfBool.shape[0],1):
            #print(i,dfBool.iloc[i].sum())
            row = dfBool.iloc[i].tolist()
            maxCheck.append(sum(row))
            pos = [j for j, x in enumerate(row) if x == 1]
            pos = [k for k in pos if k > i]
            if len(pos) != 0:
                pos.insert(0,i)
                pos = [dfBool.index[k] for k in pos]
                tempDf = pd.DataFrame([pos])
                resultDf = pd.concat([resultDf,tempDf])
        print(max(maxCheck))
        resultDfMelt = pd.melt(resultDf,id_vars=[0]).sort_values(0).dropna()
        resultDf.to_csv('DupeCAS.csv',index=False)
        resultDfMelt.to_csv('DupeCASMelt.csv',index=False)

    def calcWeightAverageDf(self,df = None,calcMethod ='weightedAverage'):
        if df is None:
            data = np.random.randint(1,200, size=200)
            data = np.reshape(data,(20,10))
            df = pd.DataFrame(data,columns=['a','b','c','d','e','f','g','h','i','j'])
            print(df)
        else:
            pass
        df = df.drop(['Fish_Count','Fish_Substances','Daphnia_Count','Daphnia_Substances','Algae_Count','Algae_Substances','Fish_Peak','Algae_Peak','Daphnia_Peak'],axis=1)
        df = df.iloc[:,0:23]
        columns1 = df.iloc[0, 0:3].drop('similarStructureCAS').index.tolist()
        print(columns1)
        columns2 = df.iloc[0,4:23].index.tolist()
        print(columns2)
        columns = columns1+columns2
        resultDf = pd.DataFrame(columns=columns)

        if calcMethod == 'weightedAverage':
            for name in df['targetCAS'].unique().tolist():
                tempDf = df[df['targetCAS'] == name]
                tempDf = tempDf[tempDf['TanimotoValue'] >= 0.6]
                calcDf =tempDf.iloc[:,5:24]
                print('DF',calcDf)
                for i in np.arange(0,len(calcDf)-1,1):
                    num = tempDf['TanimotoValue'].iloc[i]
                    calc = lambda x :x * num
                    calcDf.iloc[i] = calcDf.iloc[i].map(calc)
                print(calcDf)
                weightCalcDf =calcDf.sum()/tempDf['TanimotoValue'].sum()
                targetInfo = tempDf.iloc[0,0:5].drop(['similarStructureCAS','similarStructureName'])
                tempResult = pd.concat([targetInfo,weightCalcDf])
                resultDf = resultDf.append(tempResult,ignore_index=True)
            resultDf['type'] = 'predict'
            resultDf['calcMethod'] = calcMethod

        if calcMethod == 'weightedAverageDropMax':
            for name in df['targetCAS'].unique().tolist():
                tempDf = df[df['targetCAS'] == name]
                tempDf = tempDf[tempDf['TanimotoValue'] >= 0.6]
                if tempDf['Fish_result'].max() == tempDf['Fish_result'].min():
                    #pass
                    print('not eject',tempDf['Fish_result'].max())
                else:
                    tempDf = tempDf[tempDf['Fish_result']!=tempDf['Fish_result'].max()]
                    print('eject',tempDf['Fish_result'].max())
                calcDf =tempDf.iloc[:,5:24]
                for i in np.arange(0,len(calcDf)-1,1):
                    num = tempDf['TanimotoValue'].iloc[i]
                    calc = lambda x :x * num
                    calcDf.iloc[i] = calcDf.iloc[i].map(calc)
                #print(calcDf)
                weightCalcDf =calcDf.sum()/tempDf['TanimotoValue'].sum()
                #print(tempDf.iloc[0,0:5])
                targetInfo = tempDf.iloc[0,0:5].drop(['similarStructureCAS','similarStructureName'])
                tempResult = pd.concat([targetInfo,weightCalcDf])
                resultDf = resultDf.append(tempResult,ignore_index=True)
            resultDf['type'] = 'predict'
            resultDf['calcMethod'] = calcMethod

        if calcMethod == 'weightedAverageDropLarge':
            for name in df['targetCAS'].unique().tolist():
                tempDf = df[df['targetCAS'] == name]
                targetInfo = tempDf.iloc[0, 0:4].drop(['similarStructureCAS', 'similarStructureName'])
                targetInfo = tempDf[['targetCAS','targetName']].iloc[0]
                for type in ['fish', 'Daphnia', 'Algae']:
                    print(name,type)
                    if type == 'fish':
                        tempDf2 = tempDf.drop(['Daphnia_result','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_result','Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average'],axis=1)
                        #tempDf = tempDf[tempDf['TanimotoValue'] >= 0.6]
                        #データが一つの場合は保持
                        if tempDf2['Fish_result'].max() == tempDf2['Fish_result'].min():
                            #pass
                            print('not eject',tempDf2['Fish_result'].max())
                        #それ以外は中央値の10倍以下のみ保持
                        else:
                            tempDf2 = tempDf2[tempDf2['Fish_result'] < tempDf2['Fish_result'].median()*10]
                            tempDf2 = tempDf2[tempDf['Fish_result']>0]
                            tempDf2 = tempDf2.dropna(subset=['Fish_result'])
                            print('eject',tempDf2['Fish_result'].max())
                        calcDf =tempDf2.iloc[:,5:30]
                        for i in np.arange(0,len(calcDf)-1,1):
                            num = tempDf2['TanimotoValue'].iloc[i]
                            #print(num)
                            calc = lambda x :x * float(num)
                            calcDf.iloc[i] = calcDf.iloc[i].map(calc)
                        #print(calcDf)
                        FishWeightCalcDf =calcDf.sum()/tempDf2['TanimotoValue'].sum()
                        FishWeightCalcDf['Fish_tanimoto']  = tempDf2['TanimotoValue'].max()
                        #FishWeightCalcDf = FishWeightCalcDf[[0,6]]
                    elif type == 'Daphnia':
                        tempDf2 = tempDf.drop(['Fish_result','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Algae_result','Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average'],axis=1)
                        #tempDf = tempDf[tempDf['TanimotoValue'] >= 0.6]
                        #データが一つの場合は保持
                        if tempDf2['Daphnia_result'].max() == tempDf2['Daphnia_result'].min():
                            #pass
                            print('not eject',tempDf2['Daphnia_result'].max())
                        #それ以外は中央値の10倍以下のみ保持
                        else:
                            tempDf2 = tempDf2[tempDf2['Daphnia_result'] < tempDf2['Daphnia_result'].median()*10]
                            tempDf2 = tempDf2[tempDf['Daphnia_result']>0]
                            tempDf2 = tempDf2.dropna(subset=['Daphnia_result'])
                            #print('eject',tempDf2['Daphnia_result'].max())
                        calcDf =tempDf2.iloc[:,5:30]
                        for i in np.arange(0,len(calcDf)-1,1):
                            num = tempDf2['TanimotoValue'].iloc[i]
                            calc = lambda x :x * num
                            calcDf.iloc[i] = calcDf.iloc[i].map(calc)
                        #print(calcDf)
                        DaphniaWeightCalcDf =calcDf.sum()/tempDf2['TanimotoValue'].sum()
                        try:
                            DaphniaWeightCalcDf['Daphnia_tanimoto'] = tempDf2['TanimotoValue'].max()
                        except:
                            DaphniaWeightCalcDf['Daphnia_tanimoto'] =''
                        #DaphniaWeightCalcDf = DaphniaWeightCalcDf[[0,6]]
                        #DaphniaWeightCalcDf=   DaphniaWeightCalcDf.drop(['TanimotoValue'])

                    elif type == 'Algae':
                        #print(tempDf.head(1))
                        tempDf2 = tempDf.drop(['Fish_result','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_result','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average'],axis=1)
                        #tempDf = tempDf[tempDf['TanimotoValue'] >= 0.6]
                        #データが一つの場合は保持
                        if tempDf2['Algae_result'].max() == tempDf2['Algae_result'].min():
                            #pass
                            print('not eject',tempDf2['Algae_result'].max())
                        #それ以外は中央値の10倍以下のみ保持
                        else:
                            tempDf2 = tempDf2[tempDf2['Algae_result'] < tempDf2['Algae_result'].median()*10]
                            tempDf2 = tempDf2[tempDf['Algae_result']>0]
                            tempDf2 = tempDf2.dropna(subset=['Algae_result'])
                            #print('eject',tempDf2['Algae_result'].max())
                        calcDf =tempDf2.iloc[:,5:30]
                        for i in np.arange(0,len(calcDf)-1,1):
                            num = tempDf2['TanimotoValue'].iloc[i]
                            calc = lambda x :x * num
                            calcDf.iloc[i] = calcDf.iloc[i].map(calc)
                        #print(calcDf)
                        AlgaeWeightCalcDf =calcDf.sum()/tempDf2['TanimotoValue'].sum()
                        try:
                            AlgaeWeightCalcDf['Algae_tanimoto'] = tempDf2['TanimotoValue'].max()
                        except:
                            AlgaeWeightCalcDf['Algae_tanimoto'] = ''
                        #AlgaeWeightCalcDf = AlgaeWeightCalcDf[[0, 6]]
                        #AlgaeWeightCalcDf=AlgaeWeightCalcDf.drop(['TanimotoValue'])

                connectTemp =pd.concat([FishWeightCalcDf,DaphniaWeightCalcDf])
                weightCalcDf =pd.concat([ connectTemp,AlgaeWeightCalcDf])
                tempResult = pd.concat([targetInfo,weightCalcDf])
                # tempname = name +'.csv'
                # tempResult.to_csv(tempname)
                resultDf = resultDf.append(tempResult,ignore_index=True)
            resultDf['type'] = 'This method'
            resultDf['calcMethod'] = calcMethod
        targetToxDf = pd.read_csv('targetToxValues.csv',engine='python',encoding='utf-8')
        #resultDf = pd.read_csv('predict.csv',encoding='utf-8')
        targetColumsList = targetToxDf.columns.tolist()
        predictColumnsList = resultDf.columns.tolist()
        names = set(targetColumsList) - set(predictColumnsList)
        for name in list(names):
            resultDf[name] = '-'
            #print(resultDf[name])
        targetToxDf['TanimotoValue'] = '-'
        targetToxDf['type'] = 'DB value'
        targetToxDf['calcMethod'] = '-'
        targetToxDf['Fish_result']= targetToxDf['Fish_median']
        targetToxDf['Daphnia_result'] =targetToxDf['Daphnia_median']
        targetToxDf['Algae_result']=targetToxDf['Algae_median']
        connectDf = pd.concat([targetToxDf,resultDf])
        resultDf.to_csv('tempreult.csv')
        targetToxDf.to_csv('temptargetTox.csv')
        connectDf.to_csv('tempcheck.csv')
        '''
        columns = ['calcMethod','targetCAS','targetName','type',\
                  'Fish_Count','Fish_Substances','Fish_tanimoto','Fish_result','Fish_average','Fish_max','Fish_median','Fish_min','Fish_mode',\
                   'Daphnia_Count','Daphnia_Substances','Daphnia_tanimoto','Daphnia_result','Daphnia_average','Daphnia_max','Daphnia_median','Daphnia_min','Daphnia_mode',\
                    'Algae_Count', 'Algae_Substances', 'Algae_tanimoto', 'Algae_result', 'Algae_average', 'Algae_max', 'Algae_median', 'Algae_min', 'Algae_mode',]
        '''
        columns = ['calcMethod','targetCAS','targetName','type',\
                  'Fish_Count','Fish_Substances','Fish_tanimoto','Fish_result',\
                   'Daphnia_Count','Daphnia_Substances','Daphnia_tanimoto','Daphnia_result',\
                    'Algae_Count', 'Algae_Substances', 'Algae_tanimoto', 'Algae_result',]
        connectDf =connectDf.loc[:,columns]
        connectDf = connectDf.sort_values(['targetCAS','type'])
        #names= set(predictColumnsList) -set(targetColumsList)
        connectDf.to_csv('predict.csv',index=False,encoding='utf-8')
    def forKate(self):
        KateDf = pd.read_csv('Kate_result.csv').iloc[:,0:15]
        sepData = KateDf['95% Prediction Interval']
        sepDf = sepData.str.extract('\[(?P<min>.+)\,(?P<Max>.+)\]')
        KateDf = pd.concat([KateDf,sepDf],axis=1,)
        extractKateDf = KateDf[['CAS', 'QSAR Class','Species', 'Predicted Value [mg/L]', 'Max', 'min']]
        #改行コードの削除
        extractKateDf['CAS'] = extractKateDf['CAS'].str.replace('\n', '')
        #慢性値の削除
        extractKateDf = extractKateDf[(~(extractKateDf['Species'] == 'Fish (chronic)')) & (~(extractKateDf['Species'] == 'Daphnia (chronic)')) &  (~(extractKateDf['Species'] == 'Algae (chronic)'))]
        #表記統一
        extractKateDf['Species'] = extractKateDf['Species'].replace('(.*)\(acute\)', r'\1', regex=True)
        extractKateDf['type']='KATE'
        extractKateDf =  extractKateDf.rename(columns={'CAS':'targetCAS','QSAR Class':'Class','Predicted Value [mg/L]':'value[mg/L]','class':'Class'})
        extractKateDf['Species'] = extractKateDf['Species'].replace('Fish ','Fish').replace('Algae ','Algae').replace('Daphnia ','Daphnia')
        return extractKateDf

        # #藻類、ミジンコの慢性と急性の統合、平均値の算出
        # extractKateDf['Species'] = extractKateDf['Species'].replace('(.*)\(chronic\)', r'\1', regex=True)
        # extractKateDf['Species'] = extractKateDf['Species'].replace('(.*)\(acute\)', r'\1', regex=True)
        # resultDf = pd.DataFrame(columns=extractKateDf.columns)
        # for cas in extractKateDf['CAS'].unique():
        #     tempDf =  extractKateDf[extractKateDf['CAS']==cas]
        #     for species in tempDf['Species'].unique():
        #         extractDf = tempDf[tempDf['Species']==species]
        #         if len(extractDf) == 0:
        #             calcDf = pd.DataFrame([[cas,None,None,None,None]],columns=extractKateDf.columns.tolist())
        #         elif len(extractDf) == 1:
        #             calcDf = extractDf
        #         else:
        #                 p = extractDf[ 'Predicted Value [mg/L]'].sum()/len(extractDf)
        #                 max = tempDf['Max'].astype(float).sum()/len(extractDf)
        #                 min = tempDf['min'].astype(float).sum()/len(extractDf)
        #                 calcDf = pd.DataFrame([[cas,species,p,max,min]],columns = extractKateDf.columns.tolist())
        #         resultDf = pd.concat([resultDf,calcDf])
        # print(resultDf)
        # groupKate = KateDf.groupby(['CAS','Spacies'])

    def csvToToxdata(self):
        toxValues = pd.read_csv('allData.csv',engine='python',encoding="UTF-8")
        resultDf = pd.DataFrame(columns=['CAS','variable','value'])
        testDataMeltDf = pd.read_csv(".\\backup\\result.csv")
        samples = testDataMeltDf['CAS'].unique()
        columnList = ['targetCAS','targetName','Fish_Count','Fish_Substances','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances', 'Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        sampleToxValues =pd.DataFrame(columns=columnList)
        for sample in samples:
            toxValueList = []
            top = testDataMeltDf[testDataMeltDf['CAS']==sample].sort_values('value',ascending=False).head()
            toxValuesTemp=toxValues[toxValues['CAS']==sample]
            toxValueList.extend([sample,toxValuesTemp ['化学物質名'].values[0]])
            for name in ['魚類','ミジンコ類','藻類']:
                targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階']==name]
                toxValueList.extend([len(targetToxValues['毒性値']), len(targetToxValues['化学物質名'].unique())])
                if name is '魚類' :
                    targetToxValues = targetToxValues[targetToxValues['暴露時間（日）']<5]
                else:
                    pass
                try:
                    toxValueList.extend([targetToxValues['毒性値'].median(),targetToxValues['毒性値'].min(),targetToxValues['毒性値'].max(),targetToxValues['毒性値'].mode()[0],targetToxValues['毒性値'].mean()])
                except:
                    toxValueList.extend(['nan', 'nan', 'nan', 'nan','nan'])
            print(toxValueList)
            tempDf = pd.DataFrame([toxValueList],columns=columnList)
            sampleToxValues = pd.concat([sampleToxValues,tempDf])

            resultDf = pd.concat([resultDf,top])
        resultDf.to_csv('result.csv',index=False)
        #clusterData.to_csv('clusterData.csv',index=False)
        sampleToxValues.to_csv('targetToxValues.csv',index=False,encoding='utf-8')
    def forEcosar(self):
        EcosarDf = pd.read_csv('ECOSAR_result.csv',engine='python')
        extEcosarDf = EcosarDf[(EcosarDf['End Point']=='LC50') | (EcosarDf['End Point']=='EC50')]
        extEcosarDf = extEcosarDf.drop(['Duration','Max Log Kow','Flags','End Point'],axis=1)
        extEcosarDf = extEcosarDf.query('Organism != ["Mysid (SW)","Lemna gibba","Mysid", "Earthworm"]')
        extEcosarDf = extEcosarDf.rename(columns ={'Concentrations[mg/L]':'value[mg/L]','CAS':'targetCAS'})
        extEcosarDf['type'] = 'ECOSAR'
        extEcosarDf['Species'] = extEcosarDf['Organism'].replace('(.*)\(SW\)', r'\1', regex=True).replace('Daphnid','Daphnia').replace('Green Algae','Algae').replace('Green Algae ','Algae').replace('Fish ','Fish')
        return extEcosarDf

    def forMypredict(self):
        MypredictDf = pd.read_csv('predict.csv',engine = 'python')
        extMypredictDf = MypredictDf[['targetCAS','type','TanimotoValue','Fish_max','Fish_result',	'Fish_min','Algae_max','Algae_result','Algae_min','Daphnia_max',	'Daphnia_result','Daphnia_min','Daphnia_mode']]
        FishDf = extMypredictDf[['targetCAS','type','TanimotoValue','Fish_max','Fish_result','Fish_min']]
        DaphniaDf = extMypredictDf[['targetCAS','type','TanimotoValue','Daphnia_max',	'Daphnia_result','Daphnia_min']]
        AlgaeDf =extMypredictDf[['targetCAS','type','TanimotoValue','Algae_max','Algae_result','Algae_min']]
        FishDf['Species'] = 'Fish'
        DaphniaDf['Species'] = 'Daphnia'
        AlgaeDf['Species'] = 'Algae'
        newcolumns = ['targetCAS','type','TanimotoValue','Max','value[mg/L]','min','Species']
        FishDf.columns = newcolumns
        DaphniaDf.columns = newcolumns
        AlgaeDf.columns=newcolumns
        resultDf = pd.concat([FishDf,DaphniaDf])
        resultDf = pd.concat([resultDf,AlgaeDf])
        return resultDf
    def searchSpecificName(self):
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\allData.csv",engine='python',encoding='utf8')
        df = df[['CAS', '化学物質名', '毒性値', '暴露時間（日）', '生物種', '栄養段階']]
        df2= df[df['化学物質名'].str.contains('hosphoric', na=False)]
    def checkNonspecificToxic(self):
        os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
        #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
        #df = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\allDataAcute.csv',engine='python',encoding='utf-8')
        df = pd.read_csv('extChronicData.csv',engine='python',encoding='cp932')
        df = df[['CAS', '化学物質名', '毒性値', '暴露時間（日）', '生物種', '栄養段階']]
        #ランダム選択
        dfOver20 = df['CAS'].value_counts() >20
        len(df['CAS'].unique().tolist())
        over20 = dfOver20[dfOver20 == True].index.tolist()
        import random
        random.seed(0)
        select = random.sample(over20, 50)
        exDfEpo = df[df['CAS'].isin(select)]
        #抽出例
        #CAS = df['CAS'][df['化学物質名'].str.contains('carbamic', na=False)].unique()
        # #エポキシ
        # exDfEpo = df[df['CAS'].isin(['106-88-7', '106-92-3','122-60-1','2443-39-2','2451-62-9','106-91-2','556-52-5','30171-80-3'])]
        # #プロパギル
        # exDfEpo = df[df['CAS'].isin(['105512-06-9','23504-07-6','6921-29-5'])]
        # #アリル
        # exDfEpo = df[df['CAS'].isin(['4080-31-3', '109-75-1''57-06-7','589-09-3','96-05-9','131-17-9','1745-81-9','106-92-3','107-05-1','107-18-6','1471-17-6'])]
        # #DDD
        # exDfEpo = df[df['CAS'].isin(['72-54-8'])]
        #有機リン
       #  exDfEpo = df[df['CAS'].isin(['115-86-6', '126-73-8', '141-66-2', '298-07-7', '300-76-5',
       # '34364-42-6', '62-73-7', '107-66-4', '1241-94-7', '13171-21-6',
       # '2528-36-1', '29761-21-5', '311-45-5', '3689-24-5', '470-90-6',
       # '598-02-7', '68333-79-9', '6923-22-4', '7722-88-5', '96300-97-9',
       # '10042-91-8', '107-49-3', '512-56-1', '5598-15-2', '7558-80-7',
       # '7601-54-9', '7783-28-0', '96300-95-7'])]
        # #ウレタン(カーバメート)
        # exDfEpo = df[df['CAS'].isin(['3766-81-2','55406-53-6','63-25-2','114-26-1','14324-55-1','148-18-5','1563-66-2','2032-59-9','2631-40-5','123-03-0','128-04-1','138-93-2','2032-65-7','315-18-4','3766-81-2','759-94-4','81510-83-0','1111-78-0','12407-86-2','128-03-0','15339-36-3','23504-07-6','2626-83-7','2631-40-5','31502-57-5'])]
        # #カルボスルファン
        # exDfEpo = df[df['CAS'].isin(['10605-21-7', '122-42-9', '13684-63-4', '17804-35-2','23103-98-2','55406-53-6', '72490-01-8', '137-29-1', '2595-54-2', '31502-57-5'])]
        # #ピレスロイド
        # exDfEpo = df[df['CAS'].isin(['51-03-6','333-41-5','70124-77-5','584-79-2','7696-12-0','10453-86-8','26002-80-2','52645-53-1','80844-07-1','39515-40-7','68359-37-5','79538-32-2','82657-04-3','105024-66-6'])]
        #カーバメート殺虫剤
        # exDfEpo = df[df['CAS'].isin(['22781-23-3','16752-77-5','112410-23-8',\
        #                              '161050-58-4','79127-80-3','82560-54-1',\
        #                              '65907-30-4','114-26-1','55285-14-8','63-25-2',\
        #                              '1129-41-5','116-06-3','143807-66-3','23103-98-2',\
        #                              '2631-40-5','3766-81-2','1563-66-2','59669-26-0',\
        #                              '29973-13-5','28217-97-2','2686-99-9','78-57-9'])]
        # # REMS
        # varhaar1
        #exDfEpo = df[df['CAS'].isin(['111-46-6','112-27-6','109-86-4','64-17-5','110-80-5','67-63-0','75-65-0','112-34-5',\
        #                             '111-76-2','75-09-2','107-06-2','96-18-4','78-87-5','142-28-9','71-43-2','71-55-6',\
        #                             '108-88-3','56-23-5	95-47-6','106-42-3','95-50-1','106-46-7','541-73-1','120-82-1',\
        #                             '87-61-6','108-70-3','634-66-2','634-90-2'])]
        # verhaar2
        # exDfEpo = df[df['CAS'].isin(['62-53-3','95-51-2','108-42-9','106-47-8','95-82-9','554-00-7','626-43-7','95-76-1',
        #                              '634-67-3','636-30-6','95-53-4',\
        #                              '108-44-1','106-49-0','589-16-2','95-57-8','108-43-0','120-83-2','591-35-5',\
        #                              '121-73-3','88-72-2','89-59-8','88-74-4','99-09-2','100-01-6'])]
        #exDfEpo = df[df['CAS'].isin(df['CAS'].unique())]
        #Random2
        # exDfEpo = df[df['CAS'].isin(['1918-02-1', '107-41-5', '563-12-2', '1740-19-8', '576-26-1', \
        #                              '51218-45-2','1204-21-3', '7646-85-7', '64359-81-5', '10380-28-6',\
        #                             '584-79-2', '13510-49-1','944-22-9','1014-70-6', '13138-45-9',\
        #                              '2303-17-5', '10124-43-3', '11067-81-5','78-59-1','59-50-7'])]
        #exDfEpo = df[df['CAS'].isin(['64-17-5',         '7783-06-4',         '94-09-7',         '68951-67-7',         '71-36-3',         '68-12-2',         '131-17-9',         '91465-08-6',         '2425-06-1',         '8003-34-7',         '260-94-6',         '7646-79-9',         '608-73-1',         '106-44-5',         '55-38-9',         '151-50-8',         '26225-79-6',         '108-05-4',         '91-20-3',         '877-43-0',         '7173-51-5',         '3383-96-8',         '63-25-2',         '118-96-7',         '66230-04-4',         '110-80-5',         '7681-52-9',         '177256-67-6',         '1983-10-4',         '1918-02-1',         '2893-78-9',         '10102-18-8',         '7782-50-5',         '74223-64-6',         '7758-98-7',         '6515-38-4',         '34123-59-6',         '1918-16-7','122008-85-9','13593-03-8'])]
        exDfEpo = exDfEpo[['CAS', '化学物質名', '毒性値', '暴露時間（日）', '生物種', '栄養段階']]
        exDfEpo.to_csv('verhaar.csv',encoding='utf-8')
        #names = exDfEpo['化学物質名'].unique()

        names = exDfEpo['CAS'].unique().tolist()
        #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
        #clusterDf = pd.read_csv('clusterData_Louvain_cluster_08.csv',engine='python',encoding='utf-8')
        clusterDf = pd.read_csv('louvain_8071.0_150_th0.426.csv',engine='python',encoding='utf-8')
        clusters = clusterDf[clusterDf['CAS'].isin(names)]
        exDfEpo.to_csv('epoxy.csv', encoding='utf-8')
        #谷本係数上位を抽出
        #tanimotoDf = pd.read_csv('MACCSKeys_tanimoto.csv')
        tanimotoDf = pd.read_csv('chronicMACCSKeys_tanimoto.csv')
        tanimotoDf = pd.melt(tanimotoDf, id_vars=['CAS']).dropna().sort_values('CAS')
        #普段は0.6、データが少ない場合は0.3
        tanimotoDf = tanimotoDf[tanimotoDf['value']>=0.6]
        samples = exDfEpo['CAS'].unique()
        resultDf = pd.DataFrame(columns=['CAS','variable','value'])
        for sample in samples:
            #普段は5、データが少ない場合は10
            top = tanimotoDf[tanimotoDf['CAS']==sample].sort_values('value',ascending=False).head(5)
            resultDf = pd.concat([resultDf,top])
        resultDf.to_csv('result.csv',encoding='utf-8',index=False)

        #毒性値の計算
        columnList = ['targetCAS', 'targetName', 'Fish_Count', 'Fish_Substances', 'Fish_median', 'Fish_min', 'Fish_max',
                      'Fish_mode', 'Fish_average', 'Daphnia_Count', 'Daphnia_Substances', 'Daphnia_median',
                      'Daphnia_min', 'Daphnia_max', 'Daphnia_mode', 'Daphnia_average', 'Algae_Count',
                      'Algae_Substances', 'Algae_median', 'Algae_min', 'Algae_max', 'Algae_mode', 'Algae_average']
        sampleToxValues = pd.DataFrame(columns=columnList)
        toxValues = df
        for sample in samples:
            toxValueList = []
            toxValuesTemp = toxValues[toxValues['CAS'] == sample]
            toxValueList.extend([sample, toxValuesTemp['化学物質名'].values[0]])
            for name in ['魚類', 'ミジンコ類', '藻類']:
                targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階'] == name]
                toxValueList.extend([len(targetToxValues['毒性値']), len(targetToxValues['化学物質名'].unique())])
                try:
                    targetToxValues['毒性値'].values[0]
                    toxValueList.extend(
                        [targetToxValues['毒性値'].median(), targetToxValues['毒性値'].min(), targetToxValues['毒性値'].max(),
                         targetToxValues['毒性値'].mode()[0], targetToxValues['毒性値'].mean()])
                except:
                    toxValueList.extend(['nan', 'nan', 'nan', 'nan', 'nan'])
                print(toxValueList)
            tempDf = pd.DataFrame([toxValueList], columns=columnList)
            sampleToxValues = pd.concat([sampleToxValues, tempDf])
        sampleToxValues.to_csv('targetToxValues.csv',index=False,encoding='utf-8')
    def forRMSE(self):
        #os.chdir(r"C:\OneDrive\公開\勝さん共有")
        os.chdir(r"C:\onedrive\公開\勝さん共有\カーバメート(ウレタン)\魚類、ミジンコ、藻類、")
        df = pd.read_csv('ミジンコ.csv',engine='python' )
        print(len(df['CAS'].unique()))
        resultDf = pd.DataFrame(columns=['CAS','DB value','This method','ECOSAR1','ECOSAR2','ECOSAR3'])
        for CAS in df['CAS'].unique():
            data = {'CAS': [''], 'DB value': [''], 'This method': [''], 'ECOSAR1': [''], 'ECOSAR2': [''], 'ECOSAR3': [''],'ECOSAR4': ['']}
            tempResultDf = pd.DataFrame(data=data)
            tempDf = df[df['CAS']==CAS]
            tempDf = tempDf.sort_values('value')
            tempResultDf['CAS']=CAS
            ecoCount=0
            for i in np.arange(0,6,1):
                try:
                    temp =tempDf.iloc[i]
                    print(temp['TYPE'])
                    if temp['TYPE'] =='ECOSAR':
                        if ecoCount==0:
                            tempResultDf['ECOSAR1'] = temp['value']
                            ecoCount += 1
                        elif ecoCount==1:
                            tempResultDf['ECOSAR2'] = temp['value']
                            ecoCount += 1
                        elif ecoCount==2:
                            tempResultDf['ECOSAR3'] = temp['value']
                            ecoCount += 1
                        elif ecoCount==3:
                            tempResultDf['ECOSAR4'] = temp['value']
                            ecoCount += 1
                    elif temp['TYPE'] == 'This method':
                        tempResultDf['This method']=temp['value']
                    elif temp['TYPE'] == 'DB value':
                        tempResultDf['DB value']  = temp['value']
                except:
                    pass
            resultDf = resultDf.append(tempResultDf)
            print(resultDf)
        #resultDf = resultDf.dropna(subset=['DB value'])
        #resultDf = resultDf[resultDf['DB value']!='-']
        resultDf.to_csv('ミジンコdata.csv',index=False)

        # df.columns=['CAS','type','value']
        # dfFish = df[['targetCAS','targetName','type','Fish_result']]
        # dfFish2 = dfFish.pivot_table(values='Fish_result', index=['targetCAS', 'targetName'], columns='type')
        # dfPivot = df.pivot_table(values='value', index=['CAS'], columns='type')
        #
        # dfFish3 = dfFish2.dropna(how='any')
        # dfFish3  = dfFish3[dfFish3['DB value'] <100]
        # dfFish3  = dfFish3[dfFish3['This method'] < 100]
        # dfFish3  = dfFish3[dfFish3['DB value'] >0]
        # plt.scatter(dfFish3['DB value'], dfFish3['This method'])
        # plt.show()
    def checkCluster(self):
        #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
        #Louvain法
        #df1 = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\clusterData_Louvain_cluster_08.csv",engine='python')
        #DBSCAN
        #df1 = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\fingerprint\DBSCAN2.csv",engine='python')
        #GMM
        #df1 = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\fingerprint\GMM2.csv",engine='python')
        #SpectralClusering
        df1 = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\fingerprint\spectralclustering.csv",engine='python')
        #kmeans
        df1 = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\fingerprint\kmeans2.csv",engine='python')
        tanimotoDf = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\MACCSKeys_tanimoto.csv",engine='python',index_col = 'CAS')

        taniDF = tanimotoDf[tanimotoDf.sum(axis=1)!=0]
        clusters = df1['cluster'].unique().tolist()
        import math
        tanimotoSum =0
        for cluster in clusters:
            if math.isnan(cluster) :
                pass
            elif cluster==-1:
                pass
            else:
                print(cluster)
                tempDf = df1['CAS'][df1['cluster']==cluster].tolist()
                print(tempDf)
                print(len(tempDf))
                clusterDf =taniDF[tempDf].query("CAS in @tempDf",inplace=False)
                tanimotoSum = tanimotoSum+clusterDf.sum().sum()-0.5*(len(tempDf)*(len(tempDf)-1))
        print(tanimotoSum)
    def ejectLongDay(self):
        os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
        df = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\allData.csv',engine='python',encoding='utf-8')
        df = df[['CAS', '化学物質名', '毒性値', '暴露時間（日）', '生物種', '栄養段階']]
        df2 = df[((df['栄養段階'] =='魚類') & ( df['暴露時間（日）'] <= 4)) | \
             ((df['栄養段階'] == 'ミジンコ類') & (df['暴露時間（日）']  <= 2))|\
             ((df['栄養段階'] == '藻類') & (df['暴露時間（日）'] <= 4))]
        df2.to_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\allDataAcute.csv', index=False)
        df= pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\allDataAcute.csv',engine='python')
        CAS = df['CAS'].unique()
        over20casList =[]
        for cas in CAS:
            if df[df['CAS'] == cas].shape[0] >=20:
                over20casList.append(cas)
        import random
        randomChoice = random.sample(over20casList,20)

    def toRMSE(self):
        df = pd.read_clipboard()
        tempdf = df[['CAS No.','Chemical substance','Type','Tox value [mg/L]']]
        rmse = df.pivot(index='CAS No.', columns='Type', values='Tox val')
        rmse.to_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\RMSE.csv')

    def forhalfSupervisedLearning(self):
        df1 = pd.read_csv('louvain_58315.0_892_th0.426.csv')
        df2 = pd.read_csv('extChronicData.csv',encoding='cp932')
        df2 = df2.iloc[:,0:11]
        df2 = df2[df2['毒性値']>=0]
        toxValues = pd.merge(df1,df2,on = 'CAS',how = 'outer')
        #toxValues.to_csv('allDataAndCluster.csv')
        '''クラスタ帰属物質のデータ抽出'''
        clusters = toxValues['cluster'].unique()
        columnList = ['cluster','Fish_Count','Fish_Substances','Fish_Peak','Fish_result','Fish_median','Fish_min','Fish_max','Fish_mode', 'Fish_average','Daphnia_Count','Daphnia_Substances','Daphnia_Peak','Daphnia_result','Daphnia_median','Daphnia_min', 'Daphnia_max','Daphnia_mode', 'Daphnia_average','Algae_Count','Algae_Substances','Algae_Peak', 'Algae_result','Algae_median', 'Algae_min', 'Algae_max','Algae_mode', 'Algae_average']
        sampleToxValues =pd.DataFrame(columns=columnList)
        for cluster in clusters:
            if cluster == 'nan':
                pass
            else:
                toxValueList = []
                toxValuesTemp=toxValues[toxValues['cluster']==cluster]
                toxValueList.extend([cluster,])
                for name in ['魚類','ミジンコ類','藻類']:
                    targetToxValues = toxValuesTemp[toxValuesTemp['栄養段階']==name]
                    toxValueList.extend([len(targetToxValues['毒性値']),len(targetToxValues['化学物質名'].unique())])
                    #print(toxValueList)
                    try:
                        minList =list(filter(lambda x:  x <0.01  , targetToxValues['毒性値']))
                        PointZeroToOne =list(filter(lambda x:  0.1>x >=0.01  , targetToxValues['毒性値']))
                        ZeroPointToOne =list(filter(lambda x:  1>x >=0.1  , targetToxValues['毒性値']))
                        OneToTen = list(filter(lambda x: 10 > x >=1  , targetToxValues['毒性値']))
                        TenToHun = list(filter(lambda x: 100 > x >=10  , targetToxValues['毒性値']))
                        HunToThou = list(filter(lambda x: 1000 > x >= 100, targetToxValues['毒性値']))
                        ThouToTenThou = list(filter(lambda x: 10000 > x >= 1000, targetToxValues['毒性値']))
                        maxList = list(filter(lambda x: x >= 10000, targetToxValues['毒性値']))
                        result = []
                        peakCount = 0
                        for tox in [minList,PointZeroToOne,ZeroPointToOne,OneToTen,TenToHun,HunToThou,ThouToTenThou,maxList]:
                            print(cluster,name)
                            print('tox',tox)
                            if len(result) == 0:
                                result = tox
                                temp = tox
                                peakCount = 1
                            elif len(result) > len(tox):
                                pass
                            elif len(result) < len(tox):
                                result = tox
                                temp = tox
                                peakCount = 1
                            elif len(result) == len(tox):
                                result = tox
                                temp.extend(tox)
                                peakCount += 1
                        result = temp
                        try:
                            resultValue = statistics.median(result)
                        except:
                            resultValue = 'nan'
                        print('resutlt', resultValue,resultValue)
                        if peakCount ==1 :
                            peak = 'Single Top'
                        elif peakCount == 2:
                            peak = 'Double Top'
                        elif peakCount ==3:
                            peak ='Triple Top'
                        elif peakCount > 3:
                            peak = 'Flat'
                        #print(resultValue)
                        toxValueList.extend([peak,resultValue,targetToxValues['毒性値'].median(),targetToxValues['毒性値'].min(),targetToxValues['毒性値'].max(),targetToxValues['毒性値'].mode()[0],targetToxValues['毒性値'].mean()])
                    except:
                        if len(targetToxValues['毒性値']) == 1:
                            val = targetToxValues['毒性値']
                            toxValueList.extend(['-',val, val, val, val, val, val])
                        else:
                            toxValueList.extend(['-','nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
                #print(toxValueList)
                tempDf = pd.DataFrame([toxValueList],columns=columnList)
                sampleToxValues = pd.concat([sampleToxValues,tempDf])
         sampleToxValues.to_csv('clusterTox.csv')


if __name__ == '__main__':
    #os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_data')
    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
    #df = pd.read_csv('MACCSKeys_tanimoto.csv')
    #num = 1
    num = 1
    cl = testCluster()
    if num ==1:
        cl.checkNonspecificToxic()
        # ##cl.sepTestData(df)
        # #cl.csvToToxdata()
        # # cl.mergeCalcTox()
        cl.NewMergeCalcTox()
        cl.connectNames()
        # #os.chdir(r"C:\OneDrive\公開\勝さん共有\リン酸")
        df2 = pd.read_csv('predict08withName.csv').drop(['cluster'],axis=1)
        # # #cl.calcWeightAverageDf(df2,'weightedAverageDropMax')
        cl.calcWeightAverageDf(df2,'weightedAverageDropLarge')
    else:
        cl.checkCluster()
    # df2 = pd.melt(df,id_vars=['calcMethod', 'targetCAS', 'targetName', 'type'],value_vars=['Fish_tanimoto', 'Fish_result', 'Daphnia_tanimoto', 'Daphnia_result', 'Algae_tanimoto', 'Algae_result'])
    # #cl.connectResult()
    # #cl.checkDupe(df)
    # #cl.forKate()
    # extEcosarDf = cl.forEcosar()
    # extractKateDf = cl.forKate()
    # mypredictionDf = cl.forMypredict()
    # finalDf = pd.concat([extEcosarDf,extractKateDf])
    # finalDf = pd.concat([finalDf,mypredictionDf])
    # finalDf.to_csv('predictWithKATEandECOSAR.csv',encoding='utf-8',index=False)