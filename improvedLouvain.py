#! env python
# -*- coding: utf-8 -*-

import os
import sys
import community
import networkx as nx

class improvedLouvain(object):
    def __init__(self):
    
    def louvain(self,df,cutValue=0.8):
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

if __name__ == '__main__':
    tanimotoDf = pd.read_csv(r"G:\マイドライブ\Data\tox_predict\all_data\MACCSKeys_tanimoto.csv",engine='python', index_col='CAS')

    tanimotoDF = tanimotoDf.fillna(0)
    tanimotoDf = tanimotoDf[tanimotoDf.sum(axis=1)!=0]
    CAS = tanimotoDf.index.tolist()
    tanimotoDf = tanimotoDf[CAS]
    #cl.tanimotoHist(tanimotoDf)

    LouvainDf = tanimotoDf.reset_index()
    LouvainDf.columns = range(LouvainDf.shape[1])
    df, cluster = cl.louvain(LouvainDf, cutValue)
