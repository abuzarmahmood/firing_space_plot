#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:05:03 2019

@author: abuzarmahmood
"""
import numpy as np
import matplotlib.pyplot as plt

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
    prev_weight = 0
    time_weight = []
    # Open valve
    print('Valve open for %f' % start_time)
    time_weight.append([start_time,np.float(input('Please input new weight:'))])
    time_weight.append([target_weight*time_weight[0][0]/time_weight[0][1],0])
    # Open valve
    print('Valve open for %f' % (target_weight*time_weight[0][0]/time_weight[0][1]))
    time_weight[1][1] = time_weight[1][0] - np.float(input('Please input new weight:'))
    
    stop_flag = 0
    while not stop_flag:
        weights = [x[1] for x in time_weight]
        times = [x[0] for x in time_weight]
        plt.plot(times,weights,'-o')
        plt.xlabel('Open Time (s)')
        plt.ylabel('Weight (g)')
        plt.show()
        
        min_order = np.argsort(np.abs(np.asarray(weights) - target_weight))
        m = (time_weight[np.where(min_order == 0)[0][0]][1] -  time_weight[np.where(min_order == 1)[0][0]][1]) / \
                (time_weight[np.where(min_order == 0)[0][0]][0] -  time_weight[np.where(min_order == 1)[0][0]][0])
        c = time_weight[np.where(min_order == 0)[0][0]][1] - m*time_weight[np.where(min_order == 0)[0][0]][0]
    
        time_weight.append([(target_weight - c)/m,0])
        # Open valve
        print('Valve open for %f' % (time_weight[-1][0]))
        input_weight = input('Please input new weight:')
        if len(input_weight)== 0:
            stop_flag = 1
            break
        else:
            time_weight[-1][1] = time_weight[-2][0] - np.float(input_weight)
            
    return time_weight[np.where(min_order == 0)[0]]
