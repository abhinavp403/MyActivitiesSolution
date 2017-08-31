# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: snoran

Assignment 2 : Activity Recognition

This is the solution script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from features import extract_features
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('data', 'activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 40
step_size = 20

# sampling rate should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["mean X", "mean Y", "mean Z", "var X", "var Y", "var Z", "fft1", "fft2", "histogram bin 1", "histogram bin 2", "histogram bin 3", "histogram bin 4"] # etc...
class_names = ["Stationary", "Walking"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    x = extract_features(window)
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    y = np.append(y, window_with_timestamp_and_label[10, -1])
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Plot data points
#
# -----------------------------------------------------------------------------

print("Plotting data points...")
sys.stdout.flush()
plt.figure()
formats = ['bo', 'go']
for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
    plt.plot(X[i,0], X[i,1], formats[int(y[i])])
    
plt.show()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

print("\n")
print("---------------------- Decision Tree -------------------------")

trees = [] # various decision tree classifiers 
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=3))
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=5))
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features="sqrt"))
trees.append(DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features="0.9"))

for tree_index, tree in enumerate(trees):

    total_accuracy = 0.0
    total_precision = [0.0, 0.0]
    total_recall = [0.0, 0.0]
    
    cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
    for i, (train_indexes, test_indexes) in enumerate(cv):
        X_train = X[train_indexes, :]
        y_train = y[train_indexes]
        X_test = X[test_indexes, :]
        y_test = y[test_indexes]
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
        sys.stdout.flush()
        tree.fit(X_train, y_train)
        print("Evaluating classifier over {} points...".format(len(y_test)))
        
        # predict the labels on the test data
        y_pred = tree.predict(X_test)
    
        # show the comparison between the predicted and ground-truth labels
        conf = confusion_matrix(y_test, y_pred, labels=[0,1])
        
        accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
        precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
        recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))
        
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        
        print("The accuracy is {}".format(accuracy))  
        print("The precision is {}".format(precision))    
        print("The recall is {}".format(recall))
        
        print("\n")
        sys.stdout.flush()
        
    print("The average accuracy is {}".format(total_accuracy/10.0))  
    print("The average precision is {}".format(total_precision/10.0))    
    print("The average recall is {}".format(total_recall/10.0))  
    
    print("Training decision tree classifier on entire dataset...")
    tree.fit(X, y)
    print("Saving decision tree visualization to disk...")
    export_graphviz(tree, out_file='tree{}.dot'.format(tree_index), feature_names = feature_names)

print("\n")
print("---------------------- Support Vector Machine -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0]
total_recall = [0.0, 0.0]
for i, (train_indexes, test_indexes) in enumerate(cv):
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    print("Fold {} : Training SVM classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    C = 1.0  # SVM regularization parameter
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    
    print("Evaluating classifier over {} points...".format(len(y_test)))
    # predict the labels on the test data
    y_pred = clf.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=[0,1])
    
    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))
    
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    
    print("The accuracy is {}".format(accuracy))  
    print("The precision is {}".format(precision))    
    print("The recall is {}".format(recall))
    
    print("\n")
    sys.stdout.flush()
    
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

print("Training SVM classifier on entire dataset...")
clf.fit(X, y)

# when ready, set this to the best model you found, trained on all the data:
best_classifier = RandomForestClassifier(n_estimators=100)
best_classifier.fit(X,y) 
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)