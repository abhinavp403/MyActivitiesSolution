# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:54:06 2016

@author: snoran
"""

from kafka import KafkaConsumer
from kafka import KafkaProducer
import threading
import time
import numpy as np
import json
from feature_function import compute_features
import pickle
import sys

# The consumer is responsible for receiving sensor data from Kafka
consumer = KafkaConsumer('sensor-message',
                         group_id='my-group' + str(time.time()),
                         bootstrap_servers=['none.cs.umass.edu:9092'])
                         
# The producer is responsible for sending data, e.g. notifications, to Kafka                         
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'), 
                         bootstrap_servers=['none.cs.umass.edu:9092'])
                         
with open('lr_classifier.pickle', 'rb') as f:
    lr = pickle.load(f)
    
last_detection = 0

def predict(window):
    """
    Here you will use the classifier you trained to predict 
    the current activity over the given window of sensor data. 
    
    First compute features using the same feature function you 
    used to train the classifier. Then call the classifier's 
    predict method and send the result back to the client 
    using the onActivityDetected(timetamp, activity) method.
    
    """
    X = compute_features([window[:,1:]])
    y = lr.predict(X)
    t = window[int(window_size/2),0]
    if y == 1:
        print "gesture detected" # TODO: activity type       
    else:
        print "no gesture."
        
previousMagnitude = -1000000
previousPositive = False
previousStepTimestamp = 0

def onActivityDetected(timestamp, activity):
    """
    Notifies the client of the current activity.
    """
    producer.send('sensor-message', {'user_id' : 0, 'sensor_type' : activity, 'data': {'t' : t}})
    producer.flush()
    producer.send('user_0', {'user_id' : 0, 'sensor_type' : activity, 'data': {'t' : t}})
    producer.flush()

sensor_data = []
window_size = 25 # ~1 sec
step_size = 25
last_time = 0
index = 0

for message in consumer:
    try:
        data = json.loads(message.value)
        sensor_type = data['sensor_type']
        if (sensor_type == u"SENSOR_METAWEAR_ACCEL"): #TODO: 
            t=data['data']['t']
            # if more than a second has elapsed since the last data point, clear the window:
            if t - last_time > 1000:
                index = 0
                sensor_data = []
            last_time = t
            
            x=data['data']['x']
            y=data['data']['y']
            z=data['data']['z']
            sensor_data.append((t,x,y,z))
            index+=1
            
            # make sure the data we send is not larger than the window size (due to multi-threading):
            while len(sensor_data) > window_size:
                sensor_data.pop(0)
                
            if (index >= step_size and len(sensor_data) == window_size):
                t = threading.Thread(target=predict, args=(np.asarray(sensor_data[:]),))
                t.start()
                index = 0 # reset index
            sys.stdout.flush()
    except:
        pass