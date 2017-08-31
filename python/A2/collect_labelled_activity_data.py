# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:34:11 2016

Assignment A0 : Data Collection

@author: cs390mb

This Python script receives incoming labelled accelerometer data through 
the server and saves it in .csv format to disk.

"""

import numpy as np
import os

filename="traps15.csv"

data_dir = "../data"

labelled_data = []

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

def write_data(data, send_notification):
    global labelled_data
    print "adding to data"
    
    t = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    label = 0 #data['label']
    labelled_data.append([t, x, y, z, label])      
        
def save_data_to_disk():
    global labelled_data
    d = np.asarray(labelled_data)
    np.savetxt(os.path.join(data_dir, filename), d, delimiter=",")