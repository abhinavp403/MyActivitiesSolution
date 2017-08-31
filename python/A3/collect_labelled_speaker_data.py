# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A3 : Data Collection

@author: cs390mb

This Python script receives incoming labelled audio data through 
the server and saves it in .csv format to disk.

"""

import socket
import sys
import json
import numpy as np
import os

filename="breathing_traps15.csv"

data_dir = "../data"

labelled_data = []

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

def write_data(data, send_notification):
    
    global labelled_data
    
    window = data['values']
#    print window
    label = 0 #data['label']
    window.append(label)
    labelled_data.append([window])
        
def save_data_to_disk():
    global labelled_data
    d = np.reshape(np.asarray(labelled_data), (-1,8001))
    np.savetxt(os.path.join(data_dir, filename), d, delimiter=",")