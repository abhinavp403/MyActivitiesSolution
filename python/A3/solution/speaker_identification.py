# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A0 : Data Collection

@author: cs390mb

This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label. 
The label is then sent back to the Android application via the server.

"""

import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import FeatureExtractor
import os

# Load the classifier:
A3_directory = os.path.dirname(os.path.abspath(__file__))
output_dir = 'training_output'
classifier_filename = 'classifier.pickle'

with open(os.path.join(A3_directory, output_dir, classifier_filename), 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
feature_extractor = FeatureExtractor(debug=False)

def predict(window, on_speaker_detected, *args):
    """
    Given a window of audio data, predict the speaker. 
    Then use the onSpeakerDetected(speaker) method to notify the 
    Android application. You must use the same feature 
    extraction method that you used to train the model.
    """
    
    print("Buffer filled. Making prediction over window...")
    
    # TODO: Predict class label
    
    X = feature_extractor.extract_features(np.asarray(window))
    
    print(X.shape)
    
    X = np.reshape(X,(1,-1))
    
    # Make sure labels match your training data (Erik=0, Sean=1, none=2)
    classes = ["Erik", "Sean", "None", "Soha"]
    
    index = classifier.predict(X)
    speaker = classes[int(index)]
    
    print("Speaker : {}".format(speaker))
    
    on_speaker_detected(speaker, *args)
    
    return