# -*- coding: utf-8 -*-
from rdkit.Chem import Draw
import os
import sys
import glob
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import pylab as plt
import math

class tools(object):
    def __init__(self):
        self.__ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.__EXE_PATH = sys.executable
        self.__ENV_PATH = os.path.dirname(self.__EXE_PATH)
        self.__LOG = os.path.join(self.__ENV_PATH, 'log')
    def getcsv(self):
        os.chdir(r'G:\マイドライブ\Data\tox_predict\all_data\allPictures')
        files = [os.path.abspath(p) for p in glob.glob('*.png')]
        cas = [p.replace('.png','') for p in glob.glob('*.png')]

    def main(self):
        os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    def countFiles(self,path):
        os.chdir(path)
        folders = glob.glob('*')
        #print(len(folders))
        count = []
        dirName = []
        for dir in folders:
            path1 = path + dir
            os.chdir(path1)
            files = glob.glob('*')
            #print(dir, len(files))
            count.append(len(files))
            dirName.append(dir)
            os.chdir(path)
        #dirName = [int(s) for s in dirName]
        dirName2 = []
        count2 = []
        for name,countTemp in zip(dirName,count):
            try:
                dirName2.append(int(name))
                count2.append(countTemp)
                print(name,countTemp)
            except:
                pass
        import pandas as pd
        import pylab as plt
        import seaborn as sns
        df = pd.DataFrame({"Cluster No.":dirName2,"Count":count2})
        sns.lmplot(x="Cluster No.", y="Count", data=df, fit_reg=False)
        #plt.scatter(dirName,count)
        plt.show()
    def makeClusterPictures(self,result_df,strcture=None,multi=None):
        dir_list = result_df['cluster'].unique()
        if strcture is None:
            strctureDf=pd.read_csv(
                -'G:\\マイドライブ\Data\\tox_predict\\all_data\\structure_result.csv',engine='python')
            extAllDataDf =strctureDf[['CAS','canonical_smiles']][strctureDf['CAS'].isin(strctureDf['CAS'].tolist())]
            result_df = pd.merge(result_df,extAllDataDf)
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
    def makeCAStoPictures(self,df,strcture=None):
        if strcture is None:
            strctureDf=pd.read_csv('G:\\マイドライブ\Data\\tox_predict\\all_data\\structure_result.csv',engine='python')
            extAllDataDf =strctureDf[['CAS','canonical_smiles']][strctureDf['CAS'].isin(strctureDf['CAS'].tolist())]
            df = pd.merge(df,extAllDataDf)
        try:
            os.makedirs('CAStoPictures')
        except:
            pass
        extract = zip(df['CAS'],df['canonical_smiles'])
        for CAS, smiles in extract:
            try:
                m = Chem.MolFromSmiles(smiles)
                AllChem.Compute2DCoords(m)
                name = '.\\CAStoPictures\\' + str(CAS) + '.png'
                Draw.MolToFile(m, name)
                #rdMolDraw2D(SetScale=0)
            except:
                #     print("pass1")
                pass
    def cluster_to_hist(self):
        path ='G:\\マイドライブ\\Data\\tox_predict\\all_Data\\'
        os.chdir(path)
        clusterDf = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_Data\\clusterData_Louvain_cluster_08.csv',engine='python')
        allDataDf = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_Data\\allData.csv',engine='python',encoding='utf-8')
        allDataDf =allDataDf[['CAS','栄養段階','毒性値','暴露時間（日）']]
        allDataDf =allDataDf[allDataDf['毒性値'] > 0]
        allDataClusterDf = pd.merge(allDataDf,clusterDf,how='left')
        allDataClusterDf.to_csv('G:\\マイドライブ\\Data\\tox_predict\\all_Data\\connectCluster.csv',encoding='utf-8',index=False)
        predictDf = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_Data\\predict08withName.csv',engine='python',encoding='utf-8')
        for cas in predictDf['targetCAS'].unique():
            tempDf = predictDf[predictDf['targetCAS']==cas]
            FishValues = []
            AlgaeValues=[]
            DaphniaValues= []
            for cluster in tempDf['cluster'].unique().tolist():
                if math.isnan(cluster) == True :
                    tempDf2 = tempDf[tempDf['cluster'].isnull()]
                    casNumbers = tempDf2['similarStructureCAS'].tolist()
                    print(casNumbers)
                    clusterCASDf = allDataClusterDf[allDataClusterDf['CAS'].isin(casNumbers)]
                    for type in ['魚類', 'ミジンコ類', '藻類']:
                            tempClusterCASDf = clusterCASDf[clusterCASDf['栄養段階']== type]
                            if type == '魚類':
                                fishTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=4]
                                FishValues.extend(fishTox)
                            elif type == 'ミジンコ類':
                                DaphniaTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=2]
                                DaphniaValues.extend(DaphniaTox)
                            elif type == '藻類':
                                AlgaeTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=4]
                                AlgaeValues.extend(AlgaeTox)
                else:
                    clusterCASDf = allDataClusterDf[allDataClusterDf['cluster']==cluster]
                    for type in ['魚類', 'ミジンコ類', '藻類']:
                            tempClusterCASDf = clusterCASDf[clusterCASDf['栄養段階']== type]
                            if type == '魚類':
                                fishTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=4]
                                FishValues.extend(fishTox)
                            elif type == 'ミジンコ類':
                                DaphniaTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=2]
                                DaphniaValues.extend(DaphniaTox)
                            elif type == '藻類':
                                AlgaeTox = tempClusterCASDf['毒性値'][clusterCASDf['暴露時間（日）']<=4]
                                AlgaeValues.extend(AlgaeTox)
            #Draw
            for type2 in ['魚類', 'ミジンコ類', '藻類']:
                if type2 == '魚類':
                    y = FishValues
                    name = 'fish'
                if type2 == 'ミジンコ類':
                    y = DaphniaValues
                    name = 'Daphnia'
                if type2 == '藻類':
                    y= AlgaeValues
                    name = 'Algae'
                plt.figure()
                plt.title(cas +' ' + name + ' ' + 'tox value histgram')
                plt.ylabel('count')
                plt.xlabel('tox value')
                plt.hist(y,bins=100)
                #name = '.\\' + str(type2) + '__' + str(cluster) + '__' + 'histgram.png'
                name = '.\\' + cas + '_'+ type2 +'_' + 'histgram.png'

                print(name)
                plt.savefig(name)
                #plt.show()
            # except:
            #     print(cluster)
            #     print('not Draw')
            #     pass
    def clusetrVisualize(self,df):
        import os
        os.chdir("G:\\マイドライブ\\Data\\tox_predict\\result\\newMethod")
        name = "G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\MACCSKeys.csv"
        data = pd.read_csv(name,engine='python').fillna(0)
        values = data.drop(['CAS', 'name','canonical_smiles'], axis=1).values
        from sklearn.decomposition import TruncatedSVD
        from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
        lsa = TruncatedSVD(2)
        #comp_values = lsa.fit_transform(values)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(values)
        isomap = manifold.Isomap(n_components=2).fit_transform(values)

        #compressed_center_list = lsa.fit_transform(km_model.cluster_centers_)
        data['x'] = comp_values[:, 0]
        data['y'] = comp_values[:, 1]
        compDF = data[['CAS','x','y']]
        compDF.to_csv('isomap_compData.csv',index=False)
    def scatterPlot(self):
        os.chdir(r"G:\マイドライブ\Data\tox_predict\result\newMethod")
        df = pd.read_csv('isomap_compData.csv')
        import  pylab as plt
        plt.scatter(df['x'], df['y'], c=df['cluster'], s=30)
        plt.show()
    def countAverageSimilarity(self):
        pass
    def graphForarticle(self):
        import pandas as pd
        import pylab as plt
        import seaborn as sns
        sns.set(style="darkgrid")
        #kmeans比較
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\k-means.csv",engine='python')
        # dfExtract = df.iloc[:,1:6]
        # fig = plt.figure()
        # axes = fig.subplots(ncols=5, nrows=1)
        #Ward比較
        #f = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\ward.csv",engine='python')
        # dfExtract = df.iloc[:,1:6]
        # fig = plt.figure()
        # axes = fig.subplots(ncols=5, nrows=1)
        # SSE比較
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\SSE.csv",engine='python')
        # dfExtract = df.iloc[:,1:4]
        # BIC比較
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\BIC.csv",engine='python')
        # dfExtract = df.iloc[:,1:4]
        # k-means,GMM,Ward
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\2clusterNo.csv",engine='python')
        # similalyty
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\simi.csv",engine='python')
        # # DBSCAN, Meanshift
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\DBSCANMeanShift.csv",engine='python')
         # spectral louvain
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\louvain.csv",engine='python')
        # dfExtract = df.iloc[:,1:4]
        # fig = plt.figure()
        # axes = fig.subplots(ncols=2, nrows=1)
        #Ward比較
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\diff.csv",engine='python')
        dfExtract = df.iloc[:,1:9]
        fig = plt.figure()
        axes = fig.subplots(ncols=2, nrows=1)
        colNames =dfExtract.columns.tolist()
        try:
            colNames.remove('bandwith')
        except:
            pass
        try:
            colNames.remove('Threshold(Tanimoto coefficient)')
        except:
            pass
        for ax,colName in zip(axes.ravel(),colNames):
            if colName=='DBSCAN':
                ax.scatter(x=df.iloc[:, 0], y=dfExtract[colName], s=3, label=None)
                ax.plot(df.iloc[:,0],df['DBSCAN'],linestyle='dashdot',linewidth=0.5)
            elif colName=='Mean Shift':
                ax.scatter(x=df['bandwith'], y=dfExtract[colName], s=3, label=None)
                ax.plot(df['bandwith'],dfExtract[colName], linestyle='dashdot',label=None,linewidth=0.5)
            elif colName=='Spectral Clustering':
                ax.scatter(x=df.iloc[:, 0], y=dfExtract[colName], s=3, label=None)
            elif colName=='improved Louvain method':
                ax.scatter(x=df['Threshold(Tanimoto coefficient)'], y=dfExtract[colName], s=3, label=None)
                ax.plot(df.iloc[:,2],dfExtract[colName],linestyle='dashdot',linewidth=0.5)
            else:
                ax.scatter(x=df.iloc[:,0],y=dfExtract[colName],s=3,label=None)
                ax.set_xlim(left=0, right=400)
            ax.set_title(colName)
            if colName == 'Silhouette':
                ax.set_ylabel('Silhouette coefficient',fontsize=15)
                #ax.set(ylabel='silhouette coefficient')
            elif colName == 'Elbow Method':
                #ax.set(ylabel='Distortion')
                ax.set_ylabel('Distortion',fontsize=15)
            elif colName == 'BIC':
                ax.set_ylabel('BIC',fontsize=15)
            elif colName == 'Ward':
                pass
            elif colName == 'Mean Shift':
                ax.set_xlabel('window size(bandwith)')
            elif colName == 'DBSCAN':
                ax.set_xlabel('Eps-neighborhood of a point')
            elif colName == 'improved Louvain method':
                ax.set_xlabel('Threshold(Tanimoto coefficient)')
            elif colName == 'Spectral Clustering':
                ax.set_xlabel('Number of Clusters')
                #ax.set_xlim(0,100)
                ma_10 = dfExtract[colName].rolling(window=10).mean()
                ma = ax.plot(df.iloc[:,0],ma_10,color="red", label="10MA",linewidth =0.8)
                ax.legend(loc="lower right")
            elif colName == 'Weight average Tanimoto':
                ma_5 = dfExtract[colName].rolling(window=5).mean()
                ma5 = ax.plot(df.iloc[:,0],ma_5,color='red', label="5MA",linewidth =0.6)
                ax.set_ylabel('Tanimoto coefficient',fontsize=15)
            elif colName == 'Differential coefficient':
                ma_5 = dfExtract[colName].rolling(window=5).mean()
                ma5 = ax.plot(df.iloc[:,0],ma_5,color='green', label="5MA",linewidth =0.6)
                ma_10 = dfExtract[colName].rolling(window=10).mean()
                ma10 = ax.plot(df.iloc[:,0],ma_10,color='red' ,label="10MA",linewidth =0.6)
                ax.legend(loc="lower right")
                ax.set_ylabel('differential coefficient',fontsize=15)
                ax.set_title('Differential coefficient(Weight average Tanimoto)')
            else:
                ma_5 = dfExtract[colName].rolling(window=10).mean()
                ma = ax.plot(df.iloc[:,0],ma_5,color="red", label="10MA",linewidth =0.8)
                ax.legend(loc="lower right")
            if colName == 'Silhouette':
                #pass
                ymin = 0
                ymax = 0.2
                #ymax = round(dfExtract[colName].max() +dfExtract[colName].max()/10,-1)
            elif colName == 'BIC':
                #pass
                ymin = None
                ymax = round(dfExtract[colName].max() +dfExtract[colName].max()/10,-3)
            elif colName == 'Differential coefficient':
                ymin = -0.06
                ymax = 0.06
            elif colName == 'DBSCAN':
                ymin = -1000
                ymax = 3000
            elif dfExtract[colName].max() >0:
                ymin = -1000
                #ymin = 0
                #ymax = 6000
                ymax = round(dfExtract[colName].max() +dfExtract[colName].max()/10,-2)
                #ymax = round(dfExtract[colName].max() ,-2)
            else:
                ymin = -20000
                ymax = round(dfExtract[colName].max() -dfExtract[colName].max()/10,-3)

            ax.set_ylim(bottom = ymin,top =ymax )
            ax.set_xlim(left = 0)
            #ax.set_ylim(top =ymax )
            ax.tick_params(axis='y',labelsize=6,pad=1)
            ax.tick_params(axis='x',labelsize=10,pad=1)
            #ax.xlabel("Number of Cluster", fontsize=10)
            ymax=None
            ymin=None
        #fig.text(0.5, 0.05, 'Number of Clusters', ha='center', va='center', fontsize=15)
        fig.subplots_adjust(left=0.08, right=0.95,bottom=0.15,wspace=0.2)
        fig.text(0.05, 0.5, 'Score', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.show()
    def timeGraphForarticle(self):
        import pandas as pd
        import pylab as plt
        import seaborn as sns
        sns.set(style="darkgrid")
        #kmeans比較
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\time.csv",engine='python',encoding='shift-jis')
        fig = plt.figure()
        axes = fig.subplots(ncols=6, nrows=1)
        fig.subplots_adjust(wspace=0.2)
        #clusters
        clusterDf = df.iloc[:, 0:5]
        clusterDf = clusterDf.set_index('The Number of clusters')
        clusterDf1 = clusterDf.iloc[:, 0:1]
        clusterDf2 = clusterDf.iloc[:, 1:3]
        clusterDf3 = clusterDf.iloc[:, 3:4]
        #meanshift
        MeanShiftDf =  df.iloc[:, 5:7]
        MeanShiftDf = MeanShiftDf.set_index('window size(bandwith)')
        #Louvain
        LouvainDf =  df.iloc[:, 7:9]
        LouvainDf = LouvainDf.set_index('Threshold(Tanimoto coefficient)')
        #DBSCAN
        DBSCANdf =  df.iloc[:, 9:11]
        DBSCANdf = DBSCANdf.set_index('Eps-neighborhood of a point')

        clusterDf1.plot(ax=axes[0])
        clusterDf2.plot(ax=axes[1])
        axes[1].legend(loc="upper left")
        clusterDf3.plot(ax=axes[2])
        MeanShiftDf.plot(ax = axes[3])
        DBSCANdf.plot(ax = axes[4])
        LouvainDf.plot(ax = axes[5])
        for axe in axes:
            axe.set_xlim(left = 0)
        fig.text(0.1, 0.5, 'Time [s]', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.show()
    def plotDetailofLouvain(self):
        import pandas as pd
        import pylab as plt
        import seaborn as sns
        import numpy as np
        #sns.set(style="darkgrid")
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\louvainDetail.csv",engine='python',encoding='shift-jis')
        df = df.set_index('Threshold(Tanimoto similarity)')
        fig = plt.figure()
        axes = fig.subplots(ncols=3, nrows=1)
        dfScore = df.iloc[:,[0]]
        dfTime = df.iloc[:,[1]]
        dfClusterNum = df.iloc[:,[2]]
        dfSubstances = df.iloc[:,[3]]
        dfWeight =df.iloc[:,[4]]
        dfMedian =df.iloc[:,[5]]
        dfScore.plot(ax=axes[0],legend=None)
        dfSubstances.plot(ax=axes[1],legend=None)
        dfWeight.plot(ax=axes[2],legend=None)

        ax1 = axes[0].twinx()
        ax2 = axes[1].twinx()
        ax3 = axes[2].twinx()
        #ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(axes[0].get_yticks())))
        #ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(axes[1].get_yticks())))
        dfTime.plot(ax=ax1,color='red')
        dfClusterNum.plot(ax=ax2,color='red')
        dfMedian.plot(ax=ax3,color='red')
        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = ax1.get_legend_handles_labels()
        h3, l3 = axes[1].get_legend_handles_labels()
        h4, l4 = ax2.get_legend_handles_labels()
        h5, l5 = axes[2].get_legend_handles_labels()
        h6, l6 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2,loc='center right')
        ax2.legend(h3 + h4, l3 + l4,loc='center left')
        ax3.legend(h5 + h6, l5 + l6)
        axes[0].set_xlabel('Threshold(Tanimoto similarity)')
        axes[1].set_xlabel('Threshold(Tanimoto similarity)')
        axes[2].set_xlabel('Threshold(Tanimoto similarity)')
        axes[0].set_ylabel('score')
        ax1.set_ylabel('The calculation time[s]')
        axes[1].set_ylabel('chemical substances')
        ax2.set_ylabel('The number of clusters')
        axes[2].set_ylabel('Tanimoto similarity(Weight Average)')
        ax3.set_ylabel('Tanimoto similarity(Median)')
        fig.subplots_adjust(left=0.08, right=0.95,top=1 ,bottom=0.1,wspace=0.4)
        plt.show()
    def testEdgeColor(self):
        from networkx import *
        import matplotlib.pyplot as plt
        G = Graph()
        G.add_node(1)
        # Need to specify a layout when calling the draw functions below
        # spring_layout is the default layout used within networkx (e.g. by `draw`)
        pos = spring_layout(G)
        nodes = draw_networkx_nodes(G, pos)
        # Set edge color to red
        nodes.set_edgecolor('g')
        draw_networkx_edges(G, pos)
        # Uncomment this if you want your labels
        ## draw_networkx_labels(G, pos)
        plt.show()
    def verhaarClassifer(self):
        compound = AtomAndStrcuture
        if compound is not organic:
            return 'otherClass'
        elif compound has not C,H,N,O,S,halogen:
            return  'otherClass'
        elif logP > 6 :
            return  'otherClass'
        elif MW > 600:
            return  'otherClass'
        #Class1(narcosis or baseline toxicity)
        if compound has iodine:
            pass
        elif compound has ionic group :
            pass
        elif compound has only H,C:
            return  'class1 narcosis or baseline toxicity'
        elif compound has only H,C,halogen:
            if compounds is acylic:
                if compound has halogen at B-positions from unsaturations:
                    pass
                else:
                    return 'class1 narcosis or baseline toxicity'
            elif compounds is mononyclic:
                if compound has halogens on monocyclic:
                    return 'class1 narcosis or baseline toxicity'
                elif compound has only H on monocyclic:
                    return 'class1 narcosis or baseline toxicity'
                elif compound has acylic on monocyclic or polycyclic:
                    if compound has halogen at B-positions from unsaturations:
                        pass
                    else:
                        return 'class1 narcosis or baseline toxicity'
        elif compound contain C,H,O:
        #ethers,alcohols,ketones
            if compound is benzylic alcohols or phenols:
                pass
            elif compound is epoxides or peroxides:
                pass
            elif compound is allylic or propargylic alcohols:
                pass
            elif compound is ab-unsaturated ketones:
                pass
            elif compound is 1-butenone or acetophenone:
                pass
        elif compound contain C,H,N:
            if compound is aliphatic secondary or tertiary amines:
                return 'class1 narcosis or baseline toxicity'
        elif compound contain C,H,O ,halogen:
            # ethers,alcohols,ketones
            if compound is benzylic alcohols or phenols:
                pass
            elif compound is epoxides or peroxides:
                pass
            elif compound is allylic or propargylic alcohols:
                pass
            elif compound is ab - unsaturated ketones:
                pass
            elif compound is 1 - butenone or acetophenone:
                pass
            elif compound is a or b halogen-substituted compounds:
                pass
        elif compound is more toxic than others:
            pass
    def invertColors(self):
        import glob
        import os
        import cv2
        import pylab as plt
        os.chdir(r'G:\マイドライブ\colab\allPictures')

        names = glob.glob('*.png')

        for name in names:
            #save_name= 'G:\マイドライブ\colab\invert\\' + name
            save_name='C:\\tmp\\' + name
            img = cv2.imread(name)
            inv = cv2.bitwise_not(img)
            cv2.imwrite(save_name, inv)
    def chembleMACCSkeys(self):
        import pandas as pd
        import os
        import numpy as np
        import random
        os.chdir(r"G:\マイドライブ\Data\tox_predict\chemble")
        df = pd.read_csv('inchi-smiles.csv')
        n = np.arange(0, df.shape[0], 1).tolist()
        ran = random.sample(n,120000)
        dfEx = df.iloc[ran]
        dfExSmiles = dfEx['standard_inchi']

        dfExInchi = dfEx['standard_inchi']
        dfExInchi.to_csv('extractInchi.csv',index=None)

        dfExSmiles.to_csv('extractSmiles.csv',index=None)

    def fingertPrintFromSmiles(self):
        from rdkit import Chem
        import pandas as pd
        import os
        from rdkit.Chem import MACCSkeys
        # os.chdir(r"G:\マイドライブ\Data\tox_predict\chemble")
        # df = pd.read_csv('extractSmiles.csv',header=None)

        os.chdir(r"G:\マイドライブ\Data\Meram Chronic Data")
        df = pd.read_csv('extChronicStrcture.csv',engine='python')
        df = df[['CAS', 'canonical_smiles']]
        df = df.dropna(how='any')

        #df = pd.read_csv('extractInchi.csv',header=None)
        CAS = df['CAS']
        SMILES =df['canonical_smiles']
        i = 0
        columns =np.arange(0,167,1).tolist()
        columns.insert(0, 'CAS')
        baseDf = pd.DataFrame(columns=columns)
        for cas,smiles in zip(CAS,SMILES):
            print(smiles)
            m = Chem.MolFromSmiles(smiles)
            #m = Chem.MolFromInchi(smiles)
            fgp = MACCSkeys.GenMACCSKeys(m)
            fingerprint=[]
            fingerprint.append(cas)
            for num in np.arange(0,167,1):
                num = int(num)
                if fgp.GetBit(num) == False:
                    fingerprint.append(0)
                else:
                    fingerprint.append(1)
            print(len(fingerprint))
            tempDf = pd.DataFrame([fingerprint],columns=columns)
            baseDf = pd.concat([baseDf,tempDf])
            i+=1
            print(i)
        #connect toxcity data
        #baseDf = baseDf.set_index('CAS')
        #baseDf = pd.read_csv('chronicMACCSkeys.csv')

        baseDf = baseDf.reset_index()
        toxDf = pd.read_csv('extChronicData.csv',encoding='cp932')
        toxDf = toxDf[['CAS','毒性値']]
        toxMedianDf = toxDf.groupby('CAS').median()
        toxMedianDf = toxMedianDf.reset_index()
        toxMedianDf = toxMedianDf.rename(columns={'毒性値':'toxValue'})
        targetAndMACCSDf=pd.merge(baseDf,toxMedianDf,on='CAS',how='inner')
        tox = targetAndMACCSDf['toxValue']
        logTox = np.log10(tox)
        targetAndMACCSDf['logTox'] = logTox
        targetAndMACCSDf.to_csv('chronicMACCSkeys.csv',index=False)

        tox = targetAndMACCSDf['toxValue']
        logTox = np.log10(tox)
        plt.hist(logTox, bins=500)


if __name__ == '__main__':
    tool=tools()

    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\DBSCAN\\'
    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\k-means\\'
    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\GMM\\'
    path = 'G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint\\louvain075\\'
    tool.countFiles(path)
    # os.chdir('G:\\マイドライブ\\Data\\tox_predict\\result\\fingerprint')
    # #df = pd.read_csv('G:\\マイドライブ\\Data\\tox_predict\\all_data\\clusterData_Louvain_cluster_08.csv',engine='python')
    # #tool.makeClusterPictures(df,strcture=None)
    # os.chdir('G:\\マイドライブ\\Data\\tox_predict\\all_Data')
    # df=pd.read_csv('predict08withName.csv',engine='python')
    # df = df.rename(columns={'targetCAS':'CAS'})
    # tool.makeCAStoPictures(df,strcture=None)
    #tool.cluster_to_hist()