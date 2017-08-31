# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A1 : Step Detection (SOLUTION)

@author: cs390mb

This Python script receives incoming accelerometer data through the 
server, detects step events and sends them back to the server for 
visualization/notifications.

Refer to the assignment details at ... For a beginner's 
tutorial on coding in Python, see goo.gl/aZNg0q.

"""

import numpy as np

# <SOLUTION A1>
# Define all variables used for the step detection algorithm
previousMagnitude = -1000000
previousPositive = False
previousStepTimestamp = 0
# </SOLUTION A1>

def detect_steps(data, onStepDetected):
    """
    Accelerometer-based step detection algorithm.
    
    In assignment A1, you will implement your step detection algorithm. 
    This may be functionally equivalent to your Java step detection 
    algorithm if you like. Remember to use the global keyword if you 
    would like to access global variables such as counters or buffers. 
    When a step has been detected, call the onStepDetected method, passing 
    in the timestamp, as follows:
    
        onStepDetected("STEP_DETECTED", {"timestamp" : timestamp})
        
    """
    
    # <SOLUTION A1>
    global previousMagnitude
    global previousPositive
    global previousStepTimestamp
    
    
    timestamp = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
        
    if (timestamp - previousStepTimestamp < 500):
        return
    magnitude = np.sqrt(np.sum(np.square([x, y, z])))

    if(previousMagnitude == -1000000):
        previousMagnitude = magnitude
        previousPositive = False
        
    currentPositive = magnitude > previousMagnitude
            
    if(np.abs(magnitude- previousMagnitude)>0.5):
        if(not currentPositive and previousPositive):
            previousPositive = False
            previousMagnitude = magnitude
            previousStepTimestamp = timestamp
            onStepDetected("STEP_DETECTED", {"timestamp" : timestamp})
        else:
            previousPositive = currentPositive
            previousMagnitude = magnitude
    else:
        previousPositive = currentPositive
        previousMagnitude = magnitude
    # </SOLUTION A1>
    
    return