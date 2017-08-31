# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A0 : Data Collection

@author: cs390mb

This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label. 
The label is then sent back to the Android application via the server.

"""

import sys
import os
import threading
import numpy as np
import pickle
from features import extract_features
from util import reorient, reset_vars

# Load the classifier:

A2_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(A2_directory, 'classifier.pickle'), 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()

def predict(window, on_activity_detected):
    """
    Given a window of accelerometer data, predict the activity label. 
    Then use the onActivityDetected(label) method to notify the 
    Android must use the same feature extraction that you used to 
    train the model.
    """
    
    print("Buffer filled. Making prediction over window...")
    
    # TODO: Predict class label
    X = extract_features(window)
    
    X = np.reshape(X,(1,-1))
    
    classes = ["WALK", "STATIONARY"]    
    
    print("Label : {}".format(classes[int(classifier.predict(X))]))
    
    on_activity_detected(classes[int(classifier.predict(X))])
    
    return

sensor_data = []
window_size = 20 # ~1 sec assuming 25 Hz sampling rate
step_size = 40 # no overlap
index = 0 # to keep track of how many samples we have buffered so far
reset_vars() # resets orientation variables

def buffer_and_predict(t, x, y, z, on_activity_detected):
    global index
    global sensor_data
    
    sensor_data.append(reorient(x,y,z))
    index+=1
    # make sure we have exactly window_size data points :
    while len(sensor_data) > window_size:
        sensor_data.pop(0)

    if (index >= step_size and len(sensor_data) == window_size):
        activity_recognition_thread = threading.Thread(target=predict,  args=(np.asarray(sensor_data[:]),on_activity_detected))
        activity_recognition_thread.start()
        index = 0