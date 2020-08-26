import time
import multiprocessing as mp
import RPi.GPIO as GPIO
import os
import datetime
import random
import configparser
import json
import csv

# set up raspi GPIO board.
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

# initialize nosepokes:
#rew = NosePoke(36, 11)  
light = 36
beam = 11
GPIO.setup(light, GPIO.OUT)
GPIO.setup(beam, GPIO.IN)
# initialize "reward" nosepoke. "Rew" uses GPIO pins 38 as output for the light, and 11 as
# input for the IR sensor. For the light, 1 = On, 0 = off. For the sensor, 
# 1 = uncrossed, 0 = crossed.

def flash_on(num):  # turn the light on
    GPIO.output(num, 1)

def flash_off(num):  # turn the light off
    GPIO.output(num, 0)

flash_on(light)
flash_off(light)

while True:
    time.sleep(0.01)
    print(GPIO.input(beam))

def cue_protocol(gpio_num, freq):
    freq = np.float(freq)
    while True:
        time.sleep(1/freq)
        print('OFF')
        GPIO.output(gpio_num, 0)
        time.sleep(1/freq)
        print('ON')
        GPIO.output(gpio_num, 1)

cue_protocol(light, 2)
