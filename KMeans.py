# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 01:21:07 2018

@author: Pranav
"""

#Import modules
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans 

#Points across a graph (2-D)
#(X,Y) Co Ordinates
X = np.array([[5,3],  
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [21,20],
     [10,56],
     [65,43],
     [30,35],
     [75,54],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],]) 

#See the points on a scatter graph
#plt.scatter(X[:,0],X[:,1], label='True Position')

#K-Means Clustering Model
kmeans = KMeans(n_clusters=3)  
kmeans.fit(X)

#See the centroids of the clusters
print('Cluster centroids - \n{}'.format(kmeans.cluster_centers_))

#See the clusters assigned to the points
print('\n Cluster assigned to the data points - \n {}'.format(kmeans.labels_))

#plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')

#See the clusters and the centroids
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')