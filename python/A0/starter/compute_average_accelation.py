# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:07:01 2016

@author: snoran
"""

count = 0
sum_X = 0
sum_Y = 0
sum_Z = 0

def process(timestamp, values, callback=None, *args):
    """
    Process incoming accelerometer data.
    
    You will implement this method in assignment A0. All you need to do 
    is average the incoming values along each axis and print the averages 
    to the console.
    
    You can do this by maintaining a sum variable for each axis and a counter.
    
    This method is running on its own thread, therefore if you use any global 
    variables, you must declare them outside of the method and then re-declare 
    them global within the scope of the method. For instance, if you wish to 
    modify the same sumX in all calls of this method, use
    
        global sumX
        
    This should be done within the method, but the sumX must already be defined 
    and initialized outside the method scope.
    
    To increment the counter, you can NOT use counter++. It's invalid Python 
    syntax. But you can use
        counter += 1 
    or 
        counter = counter + 1
    
    Use the print method to print to the console. You can use the + operator 
    to concatenate strings, or you can use the .format string method. Here 
    is a simple example:
    
        print("My name is {} and I am the TA for {}".format("Sean", "390MB"))

    Each set of brackets represents a replaceable value.
    
    """
    
#    print("Received data")
    # TODO: Compute the average  
    
    global count
    global sum_X
    global sum_Y
    global sum_Z
    count += 1
    sum_X += values[0]
    sum_Y += values[1]
    sum_Z += values[2]
    if (count >= 50):
        print("STEP")
        callback(sum_X, sum_Y, sum_Z, *args)
        sum_X = 0
        sum_Y = 0
        sum_Z = 0
        count = 0
    
    return