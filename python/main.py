# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:05:12 2016

@author: cs390mb

My Activities Main Python Module

This is the main entry point for the Python analytics.

You should modify the user_id field passed into the Client instance, 
so that your connection to the server can be authenticated. Also, 
uncomment the lines relevant to the current assignment. The 
map_data_to_function() function simply maps a sensor type, one of 
"SENSOR_ACCEL", "SENSOR_AUDIO" or "SENSOR_CLUSTERING_REQUEST", 
to the appropriate function for analytics.

"""

from client import Client

# TODO: uncomment each line as needed:
from A0 import compute_average_accelation
from A1 import step_detection
from A2 import activity_recognition
from A3 import speaker_identification
from A2 import collect_labelled_activity_data
from A3 import collect_labelled_speaker_data
from A5 import location_clustering

import sys

exercise = sys.argv[1]
weight = sys.argv[2]

collect_labelled_activity_data.filename = exercise + "_" + weight + ".csv"
collect_labelled_speaker_data.filename = exercise + "_" + weight + "_breathing.csv"

# instantiate the client, passing in a valid user ID:
user_id = "42.8d.7c.7d.69.92.f8.eb.dc.b5"
c = Client(user_id)

# TODO: uncomment each line as needed:
#c.map_data_to_function("SENSOR_ACCEL", compute_average_accelation.compute_average)
#c.map_data_to_function("SENSOR_ACCEL", step_detection.detect_steps)
#c.map_data_to_function("SENSOR_ACCEL", activity_recognition.buffer_and_predict)
#c.map_data_to_function("SENSOR_AUDIO", speaker_identification.predict)
#c.map_data_to_function("SENSOR_CLUSTERING_REQUEST", location_clustering.cluster)

def on_disconnect():
    collect_labelled_activity_data.save_data_to_disk()
    collect_labelled_speaker_data.save_data_to_disk()

# To collect activity data, uncomment just these next two lines:
c.map_data_to_function("SENSOR_ACCEL", collect_labelled_activity_data.write_data)
# To collect audio data, uncomment just these next two lines:
c.map_data_to_function("SENSOR_AUDIO", collect_labelled_speaker_data.write_data)
c.set_disconnect_callback(on_disconnect)

# connect to the server to begin:
c.connect()