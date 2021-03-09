#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from kmodes import kmodes
import multiprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import silhouette_score


# 模型训练不同的类别数对应的SSE及模型
def TrainCluster(df, start_k=2, end_k=20):
    print('training cluster')
    K = []
    SSE = []
    silhouette_all = []
    models = []  # 保存每次的模型
    for i in range(start_k, end_k):
        km = kmodes.KModes(n_clusters=i)
        clusters = km.fit_predict(data)
        a = metrics.silhouette_score(df, km.labels_, metric='hamming')
        SSE.append(km.cost_)  # 保存每一个k值的SSE值
        K.append(i)
        print('{} Means SSE loss = {}'.format(i, km.cost_))
        silhouette_all.append(a)
        print('这个是k={}次时的轮廓系数{}：'.format(i, a))
        b = metrics.calinski_harabasz_score(df, km.labels_)
        print('这个是k={}次时的CH指标{}：'.format(i, b))
        models.append(km)  # 保存每个k值对应的模型
    return (K, SSE, silhouette_all, models)


def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


data = pd.read_csv("1.csv")
# # # 用肘部法则来确定最佳的K值
# train_cluster_res = TrainCluster(data, start_k=2, end_k=10)
# K = train_cluster_res[0]
# SSE = train_cluster_res[1]
# plt.plot(K, SSE, '#FF8C00')
# plt.xlabel('聚类类别数k', fontproperties=getChineseFont())
# plt.ylabel('SSE', fontproperties=getChineseFont())
# plt.xticks(K)
# plt.title('用肘部法则来确定最佳的k值', fontproperties=getChineseFont())
# plt.savefig('cluster.png', dpi=300)
km = kmodes.KModes(n_clusters=3)
clusters = km.fit_predict(data)
data["cluster3"] = km.labels_
data.to_csv("among10_cluster3.csv", index=False)
# '''计算正确归类率'''
# score = np.sum(clusters[:int(len(clusters) / 2)]) + (len(clusters) / 2 - np.sum(clusters[int(len(clusters) / 2):]))
# score = score / len(clusters)
# if score >= 0.5:
#     print('正确率：' + str(score))
# else:
#     print('正确率：' + str(1 - score))

