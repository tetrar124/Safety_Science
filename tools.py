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
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\k-means.csv",engine='python')
        # dfExtract = df.iloc[:,1:6]
        # fig = plt.figure()
        # axes = fig.subplots(ncols=5, nrows=1)
        # SSE比較
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\SSE.csv",engine='python')
        # dfExtract = df.iloc[:,1:4]
        # BIC比較
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\BIC.csv",engine='python')
        # dfExtract = df.iloc[:,1:4]
        # k-means,GMM,Ward
        # df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\2clusterNo.csv",engine='python')
        # similalyty
        #df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\simi.csv",engine='python')
        # DBSCAN, Meanshift
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\DBSCANMeanShift.csv",engine='python')
        # spectral louvain
        df = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\result\newMethod\価値関数と他の手法比較\louvain.csv",engine='python')
        dfExtract = df.iloc[:,1:4]
        fig = plt.figure()
        axes = fig.subplots(ncols=2, nrows=1)
        fig.subplots_adjust(wspace=0.2)
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
            if colName=='Spectral Clustering':
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
                ax.set_xlim(0,100)
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
            ax.axis('tight')
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
                ymin = -3000
                ymax = 6000
            elif dfExtract[colName].max() >0:
                ymin = -3000
                #ymin = 0
                #ymax = 6000
                ymax = round(dfExtract[colName].max() +dfExtract[colName].max()/10,-2)

            else:
                ymin = -20000
                ymax = round(dfExtract[colName].max() -dfExtract[colName].max()/10,-3)

            ax.set_ylim(bottom = ymin,top =ymax )
            ax.set_xlim(left = 0)
            #ax.set_ylim(top =ymax )
            ax.tick_params(axis='y',labelsize=6,pad=-1)
            ax.tick_params(axis='x',labelsize=10,pad=-1)
            #ax.xlabel("Number of Cluster", fontsize=10)
            ymax=None
            ymin=None
        #fig.text(0.5, 0.03, 'Number of Clusters', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'Score', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.show()

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