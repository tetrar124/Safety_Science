#! env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,SpectralClustering,MeanShift
from sklearn.mixture import GaussianMixture
# #from rdkit.Chem import AllChem
# from rdkit import Chem
# from rdkit.Chem import rdDepictor
# from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
# from rdkit.Chem import Draw
# from rdkit import rdBase
#import skfuzzy as fuzz
import numpy as np
import time
from sklearn.metrics import silhouette_samples, silhouette_score
import math
from joblib import Parallel, delayed
from statistics import median


class clustering(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')
    def calcHAC(self,name,clusterNum = 230):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        hac_model = AgglomerativeClustering(clusterNum).fit(values)
        names = data[['CAS']]
        names['cluster'] = hac_model.labels_
        names['name'] = data['name']
        names['canonical_smiles'] = data ['canonical_smiles']
        return names
    def calcLDA(self,name,clusterNum = 230):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        lda_model =  LatentDirichletAllocation(clusterNum).fit(values)
        names = data[['CAS']]
        names['cluster'] = lda_model.labels_
        names['name'] = data['name']
        names['canonical_smiles'] = data ['canonical_smiles']
        return names
    def calcKmeans(self,name,clusterNum = 230):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        kmeans = KMeans(clusterNum, random_state=0).fit(values)
        #kmeans.cluster_centers_
        #SSE = kmeans.inertia_
        # if len(np.unique(kmeans.labels_) )==1 :
        #     silhouette_avg = 0
        # else:
        #     silhouette_avg = silhouette_score(values,kmeans.labels_)
        #print('silhouette_avg:', silhouette_avg)
        names = data[['CAS']]
        names['cluster'] = kmeans.labels_
        names['name'] = data['name']
        names['canonical_smiles'] = data ['canonical_smiles']
        #names.to_csv("kmeans.csv")
        return names
        #return names,SSE,silhouette_avg

    def calcDbscan(self,name,epsBase):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        eps = epsBase
        dbscan = DBSCAN(eps, min_samples=2).fit(values)
        names = data[['CAS']]
        names['cluster'] = dbscan.labels_
        names['name'] = data['name']
        names['canonical_smiles'] = data ['canonical_smiles']
        #names.to_csv("DBSCAN.csv")
        return names
    def calcGMM(self,name,clusterNum = 400):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        calcgmm = GaussianMixture(clusterNum,covariance_type='full').fit(values)
        gmmTarget = calcgmm.predict(values)
        gmmTargetProba = calcgmm.predict_proba(values)
        #print(gmmTarget)
        #print(gmmTargetProba.shape)

        probaDf = pd.DataFrame(gmmTargetProba)
        names = data[['CAS']]
        names['cluster'] = gmmTarget
        names['name'] = data['name']
        names['canonical_smiles'] = data ['canonical_smiles']
        names = pd.concat([names,probaDf],axis=1)
        #names.to_csv("GMM.csv")
        return names

    def calcGMMwithBIC(self,name,maxClusterNum = 15):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        lowest_BIC = np.infty
        changeBIC = []
        for clusterNum in np.arange(1,maxClusterNum,1):
            calcgmm = GaussianMixture(clusterNum,covariance_type='full')
            calcgmm.fit(values)
            BIC = calcgmm.bic(values)
            print(clusterNum)
            print('BIC:%d' % BIC)
            changeBIC.append(BIC)
            if lowest_BIC > BIC:
                lowest_BIC = BIC
                gmmTarget = calcgmm.predict(values)
                gmmTargetProba = calcgmm.predict_proba(values)
                print(gmmTarget)
                print(gmmTargetProba.shape)

        probaDf = pd.DataFrame(gmmTargetProba)
        print(probaDf.head())
        names = data[['CAS', 'name','canonical_smiles']]
        names['cluster'] = gmmTarget
        names = pd.concat([names,probaDf],axis=1)
        names.to_csv("GMM.csv")
        print(changeBIC)
        data = np.array(changeBIC)
        np.savetxt('changeBIC.csv',data, delimiter=',')

        return names

    def calcFussyCMean(self,name):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name).fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            values.T, 30, 0.5, error=0.005, maxiter=1000, init=None)
        columns = np.arange(0,len(u0[0]),1)
        index = np.arange(0,len(u0),1)
        df = pd.DataFrame(u0,columns=columns,index=index).T
        names = data[['CAS', 'name','canonical_smiles']]
        result = pd.merge(names,df,left_index=True,right_index=True)
        result = result.set_index("CAS")
        result.to_csv("fuzzy.csv")
        return result
    def calcMeanShift(self,name,bandwith=1):
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint")
        data = pd.read_csv(name, engine='python').fillna(0)
        values = data.drop(['CAS', 'name', 'canonical_smiles'], axis=1).values
        print('start')
        calcmean = MeanShift(bandwith, n_jobs=-1).fit(values)
        print('fitting')
        meanShiftTarget = calcmean.predict(values)
        print('next')

        probaDf = pd.DataFrame(meanShiftTarget)
        names = data[['CAS']]
        names['cluster'] = meanShiftTarget
        names['name'] = data['name']
        names['canonical_smiles'] = data['canonical_smiles']
        names = pd.concat([names, probaDf], axis=1)
        # names.to_csv("GMM.csv")
        return names
    def coor_csv2cluster(self,df,nClusters):
        # df2 = df
        # n = df2.shape[0]
        # index = df2.iloc[:, 0]
        # df = df2.drop(0, axis=1)
        # df = df.iloc[:n, :n]
        # df = df.astype('float')
        # df[df == 1] = 0
        #
        # for i in np.arange(0, n, 1):
        #     for j in np.arange(0, n, 1):
        #         if i == j:
        #             break
        #         else:
        #             df.iat[i, j] = df.iat[j, i]
        # plt.subplot(3, 1, 2)
        # sns.heatmap(df, fmt='g', cmap="coolwarm")
        # # sns.heatmap(df)
        values = df.values
        spectral = SpectralClustering(nClusters, eigen_solver='arpack',affinity='precomputed')
        result = spectral.fit(values)
        y_pred = result.labels_.astype(np.int)
        resultDf = pd.DataFrame({'CAS':df.index,'cluster':y_pred})
        return resultDf
        def testPlot(y_pred):
            df2 = pd.DataFrame(y_pred)
            df3 = df2.sort_values(by=[0], ascending=True)
            index = index.iloc[df3.index]
            print(index.iloc[df3.index])
            df = df.iloc[:, df3.index]
            df = df.iloc[df3.index, :]
            df = df.set_index(index)
            df.columns = index
            print(df3.index)
            plt.subplot(3, 1, 3)
            sns.heatmap(df, fmt='g', cmap="coolwarm")
            df3 = df3.set_index(index)
            df['cluster'] = df3
            # sns.heatmap(df)
            save_df = df['cluster']
            save_df.to_csv('spectral.csv')
            plt.show()

    def louvain(self,df,cutValue=0.8):
        import community
        import networkx as nx
        df2 = df
        n = df2.shape[0]
        index = df2.iloc[:, 0]
        df = df2.drop(0, axis=1)
        df = df.iloc[:n, :n]
        df = df.astype('float')
        df[df == 1] = 0
        # seaborn.heatmap(df)
        # plt.show()
        Graph = nx.Graph()
        #print(np.show_config())
        for i, col in enumerate(df.columns):
            names = np.repeat(df.index[i], df.index.shape[0])
            pair = list(zip(names, df.index, df[col]))
            result = []
            node1 = []
            node2 = []
            for j, a in enumerate(pair):
                weight = float(a[2])
                # if weight > 0:
                if weight < cutValue:
                    pass
                else:
                    #weight = weight * 30
                    node1.append(a[0])
                    node2.append(a[1])
                    result.append((a[0], a[1], weight))
            node1 = set(node1)
            node2 = set(node2)
            nodes = list(node2 | node1)
            Graph.add_nodes_from(nodes)
            Graph.add_weighted_edges_from(result)
        nodes = Graph.nodes()
        #print(Graph.edges(data=True))
        cas_numbers = []
        for cas_index in nodes:
            cas_numbers.append(index[cas_index])
        partition = community.best_partition(Graph,weight='weight')
        #partition = community.best_partition(Graph)
        pos = nx.spring_layout(Graph, k=0.5)
        num = list(partition.keys())
        cluster_list = []
        print(len(partition.keys()))
        for n in num:
            cluster_list.append(partition[n])
        print(len(index), len(cluster_list))
        result_df = pd.DataFrame({'CAS': cas_numbers, 'cluster': cluster_list})
        #result_df = result_df.set_index('CAS')
        #result_df.to_csv('clusterData_Louvain_cluster_08.csv')
        # result_df.to_csv('Louvain_cluster_075.csv')
        if len(cluster_list) ==0:
            maxCluster_list =0
        else:
            maxCluster_list = max(cluster_list)
        return result_df,maxCluster_list

    def makePictures(self,result_df,multi=None):
        dir_list = result_df['cluster'].unique()
        try:
            os.makedirs('pics')
        except:
            pass
        os.chdir('.\\pics')
        for dir in dir_list:
            dir = str(dir)
            try:
                os.makedirs(dir)
            except:
                pass
        extract = zip(result_df['CAS'], result_df['canonical_smiles'], result_df['cluster'])

        for CAS, smiles, cluster in extract:
            try:
                m = Chem.MolFromSmiles(smiles)
                AllChem.Compute2DCoords(m)
                name = '.\\' + str(cluster) + '\\' + str(CAS) + '.png'
                #if str(tox_median) == 'nan':
                    #print(name)
                Draw.MolToFile(m, name)
            except:
                #     print("pass1")
                pass
    def checkCluster(self,df1,tanimotoDf,threshold):
        taniDF = tanimotoDf[tanimotoDf.sum(axis=1)!=0]
        clusters = df1['cluster'].unique().tolist()
        #startTime=time.time()
        r = Parallel(n_jobs=-1,backend="threading")([delayed(self.checkClusterInner)(cluster, df1, taniDF) for cluster in clusters])
        print(r)
        try:
            single = r.count(0)
        except:
            single = 0
        if len(r) == 0:
            return 0
        else:
            return sum(r),single
        # tanimotoSum =0
        # for i,cluster in enumerate(clusters):
        #     #tanimotoSum.append(self.chekClusterInner(cluster,df1,taniDf))
        #     if math.isnan(cluster) :
        #         pass
        #     elif cluster==-1:
        #         pass
        #     else:
        #         #print(cluster)
        #         tempDf = df1['CAS'][df1['cluster']==cluster].tolist()
        #         #print(tempDf)
        #         #print(len(tempDf))
        #         clusterDf =taniDF[tempDf].query("CAS in @tempDf",inplace=False)
        #         tanimotoSum = tanimotoSum+clusterDf.sum().sum()-threshold*(len(tempDf)*(len(tempDf)-1))
        #         #tanimotoSum = tanimotoSum+(clusterDf.sum().sum()-0.5*(len(tempDf)*(len(tempDf)-1))/max(clusters)
        # endTime = time.time() - startTime
        # print(endTime,tanimotoSum)
        # print(i)
        # return tanimotoSum

    def checkClusterInner(self,cluster,df1,taniDF):
            if math.isnan(cluster) :
                pass
            elif cluster==-1:
                pass
            else:
                #print(cluster)
                tempDf = df1['CAS'][df1['cluster']==cluster].tolist()
                #print(tempDf)
                #print(len(tempDf))
                clusterDf =taniDF[tempDf].query("CAS in @tempDf",inplace=False)
                tanimotoVal =clusterDf.sum().sum()-threshold*(len(tempDf)*(len(tempDf)-1))
            return tanimotoVal
    def calcAverageTanimoto(self, df1, tanimotoDf, threshold):
        taniDF = tanimotoDf[tanimotoDf.sum(axis=1) != 0]
        clusters = df1['cluster'].unique().tolist()
        #startTime = time.time()
        r = Parallel(n_jobs=-1, backend="threading")([delayed(self.checkAverageTanmioto)(cluster, df1, taniDF) for cluster in clusters])
        #print(r)
        if len(r) == 0:
            return 0,0,0,0
        else:
            rTempDf = pd.DataFrame(r, columns=('taniFunc', 'taniWeight', 'taniVal', 'members','size'))
            #rTempDf.to_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\forCheck.csv")
            members = [i for i in rTempDf['members'].tolist() if i != 1]
            tempSize = rTempDf['size'].tolist()
            size = sum([i for i in rTempDf['size'].tolist() if i != 0])
            forMedian = [i for i in rTempDf['taniVal'].tolist() if i != 0]
            return rTempDf['taniFunc'].sum(),rTempDf['taniWeight'].sum()/size,np.median(forMedian),sum(members)

    def checkAverageTanmioto(self, cluster, df1, taniDF):
        if math.isnan(cluster):
            return 0, 0, 0, 0 ,0
        elif cluster == -1:
            return 0, 0, 0, 0 , 0
        else:
            tempDf = df1['CAS'][df1['cluster'] == cluster].tolist()
            clusterDf = taniDF[tempDf].query("CAS in @tempDf", inplace=False)
            if len(tempDf) == 1:
                return 0, 0, 0, 1, 0
            else:
                tanimotoWeightVal = clusterDf.sum().sum()
                members = len(tempDf)
                if members == 1:
                    size = 0
                else:
                    size = members* (members-1)
                tanimotoFunction =tanimotoWeightVal-threshold*size
                if size == 0:
                    tanimotoVal = 0
                else:
                    tanimotoVal = tanimotoWeightVal /size
                return tanimotoFunction,tanimotoWeightVal,tanimotoVal,members,size

    def calcStatVal(self):
        tanimotoDf = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\MACCSKeys_tanimoto.csv", engine='python',index_col='CAS')
        tanimotoDf = tanimotoDf.fillna(0)
        values = tanimotoDf.values
        values = values[values!=0]
        median = np.median(values)
        from scipy import stats
        stats.scoreatpercentile(values, 95.5)

    def tanimotoHist(self,tanimotoDf):
        import pylab as plt
        zeros = int(tanimotoDf.shape[0]*tanimotoDf.shape[0]/2)+1 +tanimotoDf.shape[0]
        dataCount = (tanimotoDf.shape[0]*tanimotoDf.shape[0])/zeros
        values = np.triu(tanimotoDf.values, k=1).flatten().tolist()
        plotData = []
        i = 1
        for value in values:
            if value == 0:
                if i < zeros:
                    i += 1
                else:
                    plotData.append(0)
            elif value > 0:
                a,_ = divmod(value,0.005)
                v = (a * 0.005 + 0.005)
                plotData.append(round(v, 3))
        fig, ax1 = plt.subplots()
        ax1.hist(plotData,bins=100,density=True)
        ax2 = ax1.twinx()
        ax2.hist(plotData,bins=100,range = (0,1),density=True,cumulative = True, histtype="step",color='r',linestyle="dotted")
        ax_yticklocs = ax1.yaxis.get_ticklocs()
        #ax_yticklocs = list(map(lambda x: x * len(range(0,1))* 1.0 /100, ax_yticklocs))
        ax_yticklocs = list(map(lambda x: x * len(range(0,1))* 1.0 , ax_yticklocs))
        ax1.yaxis.set_ticklabels(list(map(lambda x: "%0.2f" % x, ax_yticklocs)))
        plt.xlim([0,1])
        ax1.set_xlabel('Tanimoto Similarity')
        ax2.set_ylabel('Cumulative Probability[%]',color='r')
        ax1.set_ylabel('Probability Density[%]')
        plt.show()

    def filesearch(self):
        import  glob
        for name in glob.glob('*.py'):
            with open(name, encoding="utf-8") as f:
                a = f.read()
                # print(a)
                # print(a.find('hist'))
                if a.find('Prob') > 0:
                    print(name)

if __name__ == '__main__':
    cl = clustering()
    #name = r'G:\マイドライブ\Data\Meram Chronic Data\cembleChronicMACCSKeys_tanimoto.csv'
    #name = "G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\MACCSKeys.csv"
    tanimotoDf = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\MACCSKeys_tanimoto.csv", engine='python',                           index_col='CAS')
    tanimotoDf = pd.read_csv(name, engine='python', index_col='CAS')
    #df = cl.calcKmeans(name)
    #df = cl.calcFussyCMean(name)
    #df = cl.calcDbscan(name)
    #df = cl.calcGMM(name)
    #spectralcsv = "C:\\googledrive\\Data\\tox_predict\\result\\fingerprint\\MACCSKeys.csv"
    #cl.makePictures(df)
    #cl.calcGMMwithBIC(name)
    #tanimotoDf = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\MACCSKeys_tanimoto.csv", engine='python',index_col='CAS')
    tanimotoDF = tanimotoDf.fillna(0)
    tanimotoDf = tanimotoDf[tanimotoDf.sum(axis=1)!=0]
    CAS = tanimotoDf.index.tolist()
    tanimotoDf = tanimotoDf[CAS]
    #cl.tanimotoHist(tanimotoDf)

    LouvainDf = tanimotoDf.reset_index()
    LouvainDf.columns = range(LouvainDf.shape[1])
    #for method in ['meanshift']:
    for method in ['louvain']:
    #for method in ['kmeans']:
    #for method in ['DBSCAN']:
    #for method in ['HAC']:
    #for method in ['spectral']:
    #for method in ['GMM','louvain','spectral','kmeans','HAC','DBSCAN','meanshift']:
    #for method in ['HAC','GMM']:
        # 68.2percentile(1σ)
        # threshold = 0.214
        # 95.5percentile(2σ)
        # threshold = 0.426
        # 99.7percentile(3σ）
        # threshold = 0.738
        #threshold = 0.5
        #for threshold in [0.426,0.738,0.5,0.6,0.214]:
        #for threshold in [0.5,0.6,0.214]:
        for threshold in [0.426]:
        #for threshold in [0.738,0.5,0.214]:
            tanimotoMax = - np.inf
            tempMaxDf = None
            tempClusterNum = None
            tanimotoChange = []
            cluster_change = []
            times = []
            substances =[]
            SSEs = []
            silhouettes =[]
            medians = []
            weightMean=[]
            if method == 'louvain' :
                #setRange =  np.arange(0,1,0.002)
                setRange = [0.83]
                setRange = [0.83]
                #setRange = np.arange(0.8, 0.6, -0.01)
            elif method == 'DBSCAN' :
                setRange = np.arange(0.01,2.01,0.01)
            elif method == 'meanshift':
                setRange = np.arange(0.04, 4, 0.04)
            else :
                setRange = np.arange(1, 401, 1)
            for i,cluster in enumerate(setRange):
                print(cluster)
                start = time.time()
                #MeanShift
                if method == 'meanshift':
                    df = cl.calcMeanShift(name,cluster)
                #Louvain
                if method == 'louvain':
                    df ,cluster =cl.louvain(LouvainDf,cluster)
                #spectral clustering
                elif method =='spectral':
                    df = cl.coor_csv2cluster(tanimotoDf,cluster)
                #HAC
                elif method =='HAC':
                    df = cl.calcHAC(name,cluster)
                #GMM
                elif method == 'GMM':
                    df = cl.calcGMM(name,cluster)
                #DBSCAN
                elif method == 'DBSCAN':
                    df = cl.calcDbscan(name,clusterc)
                #kmeans
                elif method == 'kmeans':
                    df = cl.calcKmeans(name,cluster)
                    #df ,SSE ,silhouette_avg=  cl.calcKmeans(name,cluster)
                    #silhouettes.append(silhouette_avg)
                    #SSEs.append(SSE)
                elapsed_time = time.time() - start
                print('time cost')
                print(elapsed_time)
                times.append(elapsed_time)
                cluster_change.append(cluster)
                #tanimotoSum ,single = cl.checkCluster(df,tanimotoDf,threshold)
                #mean,median
                #df.to_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\basedf.csv")
                tanimotoSum,weightTanimoto,median,members = cl.calcAverageTanimoto(df, tanimotoDf, threshold)
                substances.append(members)
                tanimotoChange.append(tanimotoSum)
                medians.append(median)
                weightMean.append(weightTanimoto)
                #print(i)
                #substances.append(df.shape[0])
                if tanimotoSum > tanimotoMax :
                    tanimotoMax = tanimotoSum
                    tempMaxDf = df
                    tempClusterNum = cluster
            #os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\newMethod")
            os.chdir('G:\マイドライブ\Data\Meram Chronic Data')
            import datetime
            t = datetime.datetime.now()
            timestamp = str(t.year) + str(t.month) + str(t.day) + str(t.hour) + str(t.minute)
            tanimotoChangeCsv = str(method) + '_' + 'shre_' + str(threshold) + '_tanimotoChange.csv'
            if len(SSEs) !=0:
                tempDf = pd.DataFrame({'cluster':cluster_change,'checkValues':tanimotoChange,'calcTime':times,'SSE':SSEs,'silhouettes':silhouettes,'substances':substances})
                tempDf.to_csv(tanimotoChangeCsv, sep=',', index=False)
            elif median != None:
                tempDf = pd.DataFrame({'cluster': cluster_change,'Score': tanimotoChange,'Weight average Tanimoto':weightMean,'Tanimoto Median':medians,'substances': substances,'calcTime':times})
                tanimotoChangeCsv = str(method) + '_' + 'shre_' + str(threshold) + '_tanimotoChange' + timestamp + '.csv'
                tempDf.to_csv(tanimotoChangeCsv, sep=',',index = False)
                print('now saving')
            resultCSV = str(method) + '_' + str(round(tanimotoMax)) +'_' + str(tempClusterNum) +'_'+'th'+str(threshold)+ '.csv'
            tempMaxDf.to_csv(resultCSV, sep=',',index = False)