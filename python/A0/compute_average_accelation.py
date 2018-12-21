# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:07:01 2016

@author: snoran
"""

count = 0
sum_X = 0
sum_Y = 0
sum_Z = 0

def compute_average(data, send_notification):
    """
    Compute the average for each axis of accelerometer data every 200 samples. 
    This method is called for each sample and the data is stored in "data".
    You can access it as follows:
    
        data['x'], data['y'], data['z'] or data['t'] 
    
    for the x-, y-, z-values and time respectively.
        
    When the average is computed, send it back over the server by first wrapping 
    the averages in a Python dictionary:
    
        json = {"average_X" : x_average, "average_Y": y_average, "average_Z": z_average}
        
    Then call
    
        send_notification("AVERAGE_ACCELERATION", json)
        
    You can compute the average by maintaining sums and a counter. Since these 
    variables must be accessible with each call, you should make them global. 
    Global variables are defined outside the scope of the method. Then inside 
    the method, they can only be accessed if you first specify it as global, e.g. 

        global count
        
    The base code sends dummy values to the mobile device every 200 samples. Make sure
    this works to start.
    """
    
    global count
    
    # dummy values
    average_X = 1.5
    average_Y = 9.2
    average_Z = -3.4
    
    count += 1
    
    # <SOLUTION A0>
    global sum_X
    global sum_Y
    global sum_Z
    
    sum_X += data['x']
    sum_Y += data['y']
    sum_Z += data['z']
    # </SOLUTION A0>
    if (count >= 200):
        # <SOLUTION A0>
        average_X = sum_X / count
        average_Y = sum_Y / count
        average_Z = sum_Z / count
        # </SOLUTION A0>
        # send_notification is a method handle, make sure that it is not None (equivalent to False in an if):
        if send_notification:
            send_notification("AVERAGE_ACCELERATION", {"average_X" : average_X, "average_Y": average_Y, "average_Z": average_Z})
        # reset vars
        count = 0
        # <SOLUTION A0>
        sum_X = 0
        sum_Y = 0
        sum_Z = 0
        # </SOLUTION A0>