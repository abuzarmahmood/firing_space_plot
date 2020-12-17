#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:05:04 2019

@author: abuzarmahmood

Calculates required open-time based on iterative linear interpolation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Import things for running pi codes
import time
from math import floor
import random
import RPi.GPIO as GPIO

# Import other things for video
from subprocess import Popen
import easygui
import numpy as np
import os

# Setup pi board
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

def autocalibrate(target_weight,start_time,outport):
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(outport, GPIO.OUT)

    data = []
    data.append((0,0))
    
    # Dispense fluid using start time
    GPIO.output(outport,1)
    time.sleep(start_time)
    GPIO.output(outport,0)

    print('Open time : %.3f sec' % start_time)
    
    # Request first data point using start time
    new_weight = np.float(input('Please input new weight, enter 0 to exit:'))
    if new_weight:
        data.append((start_time,new_weight))
    else:
        return
            
    # In loop:
    # Predict new time using 2 data points closest to target weight
    # Dispense fluid using target and request respective weight
    # Plot for visualization
    # Repeat
    
    
    fig,ax = plt.subplots()
    while True:
        weight_list = [x[1] for x in data]
        time_list = [x[0] for x in data]
        f = interpolate.interp1d(time_list, weight_list, kind='slinear')
        
        time_new_vec = np.linspace(np.min(time_list),np.max(time_list),100)
        weight_new_vec = f(time_new_vec)
        
        plt.sca(ax)
        plt.cla()
        plt.scatter(time_list,weight_list)
        plt.plot(time_new_vec, weight_new_vec)
        plt.xlabel('Open Time (s)')
        plt.ylabel('Dispensed Weight (g)')
        plt.pause(0.05)
        
        new_time_val = time_new_vec[np.argmin((np.asarray(weight_new_vec) -  target_weight)**2)]
        
        # Dispense liquid at new time val, request update to weight
        GPIO.output(outport,1)
        time.sleep(new_time_val)
        GPIO.output(outport,0)

        print('Open time : %.3f sec' % new_time_val)
        
        new_weight = np.float(input('Please input new weight, enter 0 to exit:'))
        if new_weight:
            data.append((new_time_val,new_weight))
        else:
            plt.close(fig)
            wanted_ind = np.argmin((np.asarray(weight_new_vec) -  target_weight)**2)
            return (time_new_vec[wanted_ind],weight_new_vec[wanted_ind])
        
        
    
