# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This is the solution code for extracting features over accelerometer 
windows. It doesn't include all the features mentioned in the 
assignment, but it does have mean/variance, histogram features, 
and dominant frequencies over the magnitude.

"""

import numpy as np

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)
    
def _compute_var_features(window):
    """
    Computes the variance along the x-, y- and z-axes 
    over the given window.
    """
    return np.std(window, axis=0)
    
def _compute_fft_of_magnitude_features(magnitude):
    """
    Computes the first 2 dominant frequencies of the magnitude 
    signal, given by the real-valued FFT.
    """
    return np.fft.rfft(magnitude)[:2]
    
def _compute_histogram_features(window):
    return np.histogram(window, bins=4)[0]

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature matrix X.
    
    Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    
    magnitude = np.sqrt(np.sum(np.square(window), axis=1))
    
    x = []
    
    x = np.append(x, _compute_mean_features(window))
    x = np.append(x, _compute_var_features(window))
    x = np.append(x, _compute_fft_of_magnitude_features(magnitude))
    x = np.append(x, _compute_histogram_features(window))
    
    return x