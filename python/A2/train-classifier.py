# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:40:36 2016

@author: snoran
"""

import os
import numpy as np
import pandas as pd
from feature_function import compute_features
from sklearn.linear_model import LogisticRegression
import pickle

window_size = 25 # ~1 sec
step_size = 25

data_dir = 'data'
fname = os.path.join(data_dir, "ACCEL.csv")
data1 = pd.read_csv(fname,sep=',').values

fname = os.path.join(data_dir, "ACCEL3.csv")
data2 = pd.read_csv(fname,sep=',').values

data = np.vstack((data1, data2))

windows = np.asarray([data[i:i+window_size,1:4] for i in range(0, len(data), step_size)][:-1])
y = np.asarray([data[i+window_size/2.,-1] for i in range(0, len(data)-window_size, step_size)])
X = compute_features(windows)

# initialize classifier: set paramters
lr = LogisticRegression(class_weight={1:0.05, 0:0.95})

# train classifier
lr.fit(X, y)

# save classifier
with open('lr_classifier.pickle', 'wb') as f:
    pickle.dump(lr, f) 