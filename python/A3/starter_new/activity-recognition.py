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
from features import extract_features, show_pitch

# TODO: Replace the string with your user ID
user_id = "42.8d.7c.7d.69.92.f8.eb.dc.b5" #"???"

'''
    This socket is used to send data back through the data collection server.
    It is used to complete the authentication. It may also be used to send 
    data or notifications back to the phone, but we will not be using that 
    functionality in this assignment.
'''
send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_socket.connect(("none.cs.umass.edu", 9999))

# Load the classifier:

with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
def onSpeakerDetected(speaker):
    """
    Notifies the client of the current speaker
    """
    print("Speaker is {}.".format(speaker))
    sys.stdout.flush()
    send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_SERVER_MESSAGE', 'message' : 'SPEAKER_DETECTED', 'data': {'speaker' : speaker}}) + "\n")

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label. 
    Then use the onActivityDetected(label) method to notify the 
    Android must use the same feature extraction that you used to 
    train the model.
    """
    
    print("Buffer filled. Making prediction over window...")
    
    # TODO: Predict class label
    X = extract_features(window)
    
    print(X.shape)
    
    X = np.reshape(X,(1,-1))
    
    classes = ["ERIK", "SEAN"]
    
    print("Label : {}".format(classes[int(classifier.predict(X))]))
    
#    onSpeakerDetected(classes[int(classifier.predict(X))])
    
    return
    
    

#################   Server Connection Code  ####################

'''
    This socket is used to receive data from the data collection server
'''
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.connect(("none.cs.umass.edu", 8888))
# ensures that after 1 second, a keyboard interrupt will close
receive_socket.settimeout(1.0)

msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.
    
    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip()
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id))

    try:
        message = sock.recv(256).strip()
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")
        
    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))
    
    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))


try:
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)
    
    print("Authenticating user for sending data...")
    sys.stdout.flush()
    authenticate(send_socket)
    
    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()
        
    previous_json = ''
    
#    sensor_data = []
#    window_size = 20 # ~1 sec assuming 25 Hz sampling rate
#    step_size = 40 # no overlap
#    index = 0 # to keep track of how many samples we have buffered so far
        
    speech_audio_data = []       
    
    while True:
        try:
            message = receive_socket.recv(1024).strip()
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_AUDIO"):
                    t=data['data']['t']
                    audio_buffer=data['data']['values']
                    print("Received audio data of length {}".format(len(audio_buffer)))
                    t = threading.Thread(target=predict, args=(np.asarray(audio_buffer),))
                    t.start()
#                    t = threading.Thread(target=extract_features, args=(audio_buffer,))
#                    t.start()
                    
#                    sensor_data.append(reorient(x,y,z))
#                    index+=1
#                    # make sure we have exactly window_size data points :
#                    while len(sensor_data) > window_size:
#                        sensor_data.pop(0)
                    
                    # TODO : pip install --user scikits.talkbox
#                    
#                    speech_audio_data.extend(audio_buffer)
#                    if (len(speech_audio_data) >= 88000):
##                        
##                
###                    if (index >= step_size and len(sensor_data) == window_size):
#                        t = threading.Thread(target=extract_features, args=(np.asarray(speech_audio_data),))
#                        t.start()
#                        speech_audio_data = [] # reset
###                    index = 0
                
            sys.stdout.flush()
        except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (e.message != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass
except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
finally:
    print >>sys.stderr, 'closing socket for receiving data'
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()
    
    print >>sys.stderr, 'closing socket for sending data'
    send_socket.shutdown(socket.SHUT_RDWR)
    send_socket.close()
    
#    show_pitch()