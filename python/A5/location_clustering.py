# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A5 : Location Clustering

@author: cs390mb

This Python script clusters location data using k-Means / Mean Shift 
upon request from the client.

"""

import numpy as np
from sklearn.cluster import KMeans, MeanShift

def cluster(data, send_clusters):
    """
    Clusters the given locations according to the algorithm specified.
    
    TODO: You should construct a N x 2 matrix of (latitude, longitude) pairs, 
    where N is the number of locations (= len(latitudes) = len(longitudes)).
    
    Then according to the algorithm parameter ("k_means" or "mean_shift"), 
    call the appropriate scikit-learn function. For k-means, k=args[0].
    
    Like classification algorithms, first create an instance of the clustering 
    algorithm object. Then the clustering is done using the fit() 
    function. The indexes of those points are then acquired using the 
    labels_ field.
    
    You can simply pass labels_ as a parameter to send_clusters() and the 
    Android application will receive the cluster indexes.
    
    """
    
    t = data['t']
    algorithm = data['algorithm']
    k = data['k']
    latitudes = data['latitudes']
    longitudes = data['longitudes']
    
    X = np.asarray([latitudes, longitudes]).T
    print X.shape
    if algorithm == 'k_means':
        # TODO: K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        print kmeans.labels_
        print kmeans.labels_.shape
        
        indexes = [int(i) for i in kmeans.labels_] #list(np.asarray(kmeans.labels_).astype(str))
        print indexes
        send_clusters("CLUSTER", {"indexes" : indexes})
    else:
        # TODO: Mean-Shift clustering
        mean_shift = MeanShift().fit(X)
        print mean_shift.labels_
        print mean_shift.labels_.shape
    
        indexes = [int(i) for i in mean_shift.labels_]
        print indexes
        send_clusters("CLUSTER", {"indexes" : indexes})
    
    return