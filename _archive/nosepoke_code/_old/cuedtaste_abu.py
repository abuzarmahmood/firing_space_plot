"""
Created on Mon Sep  9 14:15:36 2019

@author: Daniel Svedberg (dsvedberg@brandeis.edu)
"""
import time
import multiprocessing as mp
import RPi.GPIO as GPIO
import os
import datetime
import random
import configparser
import json
import csv


################################################################################
### SECTION 1: CLASSES ###

# NosePoke is a class that controls the physical nose poke device in the rig. 
# Input variable light = GPIO pin for
# controlling LED in the nosepoke (1 = on, 0 = off). 
# Input variable beam = GPIO pin receiving input from IR
# beam-break sensor inside nose poke device (1 = uncrossed, 0 = crossed)
class NosePoke:
    def __init__(self, light, beam):
        """
        light and beam are pin numbers
        """
        self.exit = None
        self.light = light
        self.beam = beam
        # endtime is a class-wide condition to help the program exit when the task
        # is over. Usually this variable is changed when a behavioral task program is initiated
        self.endtime = time.time() + 1200  
        GPIO.setup(self.light, GPIO.OUT)
        GPIO.setup(self.beam, GPIO.IN)

    def shutdown(self):
        print("blink shutdown")
        self.exit.set()

    def flash_on(self):  # turn the light on
        GPIO.output(self.light, 1)

    def flash_off(self):  # turn the light off
        GPIO.output(self.light, 0)

    # bink on and of at frequency hz (LED has physical limit of 3.9)
    def flash(self, hz, run):  
        while time.time() < self.endtime:
            if run.value == 1:
                GPIO.output(self.light, 1)
                time.sleep(2 / hz)
                GPIO.output(self.light, 0)
                time.sleep(2 / hz)
            if run.value == 0:
                GPIO.output(self.light, 0)
            if run.value == 2:
                GPIO.output(self.light, 1)

    def is_crossed(self):  # report if beam is crossed
        if GPIO.input(self.beam) == 0:
            return True
        else:
            return False

    # report when the animal has stayed out of the nosepoke 
    # for duration of [wait] seconds
    def keep_out(self, wait):  
        start = time.time()
        while True and time.time() < self.endtime:
            if self.is_crossed():
                start = time.time()
            elif time.time() - start > wait:
                break

    def kill(self):  # kind of useless method
        GPIO.output(self.light, 0)


# class TasteLine controls an individual taste-valve and its associated functions: 
# clearouts, calibrations, and deliveries. 
# Use TasteLine by declaring a TasteLine object, using clearout() to clear-out,
# and then running calibrate() to set the opentime value. 
class TasteLine:
    def __init__(self, valve, intanOut, opentime, taste):
        """
        valve       :: GPIO pin number corresponding to the valve controlling taste delivery
        intanOUt    :: GPIO pin number used to send a signal to our intan neural recording system
                        whenever a taste is delivered.
        opentime    :: how long the valve stays open for one single delivery
        taste       :: string containing name of the corresponding taste, 
                        used for datalogging in record()
        """
        self.valve = valve  
        self.intanOut = intanOut  
        self.opentime = opentime  
        self.taste = taste  

        # generating a tasteLine object automatically sets up the GPIO pins:
        GPIO.setup(self.valve, GPIO.OUT)
        GPIO.setup(self.intanOut, GPIO.OUT)

    def clearout(self, dur):  
        """
        delivery system, and clean out the tubes when we are done. 
        """
        GPIO.output(self.valve, 1)
        time.sleep(dur)
        GPIO.output(self.valve, 0)
        print('Tastant line clearing complete.')

    def calibrate(self, opentime, repeats):  
        """
        delivery, to ensure amount of liquid delivered is consistent from 
        session to session. calibrate() prompts
        user to input a calibration time, and then opens the valve 5 times 
        for that time, so the user can weigh out
        how much liquid is dispensed per delivery.
        """
        # Open ports
        for rep in range(repeats):
            GPIO.output(self.valve, 1)
            time.sleep(opentime)
            GPIO.output(self.valve, 0)
            time.sleep(3)

    def deliver(self):  
        """
        deliver() is used in the context of a task to open the 
        valve for the saved opentime to
        """
        # deliver liquid through the line
        GPIO.output(self.valve, 1)
        GPIO.output(self.intanOut, 1)
        time.sleep(self.opentime)
        GPIO.output(self.valve, 0)
        GPIO.output(self.intanOut, 0)

    def kill(self):
        GPIO.output(self.valve, 0)
        GPIO.output(self.intanOut, 0)

    def is_open(self):  # reports if valve is open
        if GPIO.input(self.valve):
            return True
        else:
            return False


### SECTION 2: MISC. FUNCTIONS

def record(poke1, poke2, lines, starttime, endtime, anID):
    """
    record() logs sensor and valve data to a .csv file. 
    Typically instantiated as a multiprocessing.process
    """
    now = datetime.datetime.now()
    d = now.strftime("%m%d%y_%Hh%Mm")
    localpath = os.getcwd()
    filepath = localpath + "/" + anID + "_" + d + ".csv"
    with open(filepath, mode='wb') as record_file:
        fieldnames = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4']
        record_writer = csv.writer(record_file, delimiter=',', quotechar='"', 
                                    quoting=csv.QUOTE_MINIMAL)
        record_writer.writerow(fieldnames)
        while time.time() < endtime:
            t = round(time.time() - starttime, 2)
            data = [str(t), str(poke1.is_crossed()), str(poke2.is_crossed())]
            for item in lines:
                if item.is_open:
                    valvestate = item.taste
                else:
                    valvestate = "None"
                data.append(str(valvestate))
            record_writer.writerow(data)
            time.sleep(0.005)


### SECTION 3: BEHAVIORAL TASK PROGRAMS ###

def cuedtaste():
    """
    cuedtaste is the central function that runs the behavioral task.
    """
    anID = raw_input("enter animal ID: ")
    runtime = input("enter runtime in minutes: ")
    starttime = time.time()  # start of task
    endtime = starttime + runtime * 60  # end of task
    rew.endtime = endtime
    trig.endtime = endtime
    iti = 5  # inter-trial-interval
    wait = 1  # how long rat has to poke trigger to activate
    Hz = 3.9  # poke lamp flash frequency
    # how long rat has to cross from trigger to rewarder after 
    # activating trigger/arming rewrader.
    crosstime = 10  

    # setting up parallel multiprocesses for light flashing and data logging
    rew_run = mp.Value("i", 0)
    trig_run = mp.Value("i", 0)

    rew_flash = mp.Process(target=rew.flash, args=(Hz, rew_run,))
    trig_flash = mp.Process(target=trig.flash, args=(Hz, trig_run,))
    recording = mp.Process(target=record, args=(rew, trig, lines, starttime, endtime, anID,))

    rew_flash.start()
    trig_flash.start()
    recording.start()

    # [state] controls state of task. 
    # Refer to PDF of hand-drawn diagram for visual guide
    state = 0  

    # this loop controls the task as it happens, when [endtime] is reached, 
    # loop exits and task program closes out
    while time.time() < endtime:
        while state == 0 and time.time() < endtime:  # state 0: base-state
            rew_keep_out = mp.Process(target=rew.keep_out, args=(iti,))
            trig_keep_out = mp.Process(target=trig.keep_out, args=(iti,))
            rew_keep_out.start()
            trig_keep_out.start()

            line = random.randint(0, 3)  # select random taste
            rew_keep_out.join()
            trig_keep_out.join()  # if rat stays out of both nose pokes, state 1 begins
            trig.play_tone()  # technically start of state 1
            trig_run.value = 1
            state = 1
            print("new trial")

        while state == 1 and time.time() < endtime:  # state 1: new trial started/arming Trigger
            if trig.is_crossed():  # once the trigger-nosepoke is crossed, move to state 2
                trig.kill_tone()  # stop playing white noise
                trig_run.value = 2  # trigger light goes from blinking to just on
                # taste-associated cue tone is played, but rat must stay in trigger 1 sec.
                lines[line].play_tone()  
                start = time.time()
                state = 2
                print("state 2")

        while state == 2 and time.time() < endtime:  # state 2: Trigger activated/arming Rewarder
            # if rat trips sensor for 1 sec. continuously,
            if trig.is_crossed() and time.time() > wait + start:  
                # move to state 3
                rew_run.value = 1  # blink rewarder
                trig_run.value = 0 # stop blinking trigger
                deadline = time.time() + crosstime # rat has 10 sec to activate rewarder
                start = time.time()
                state = 3
                print("state 3")

            if not trig.is_crossed():  # rat pulled out too early, return to state 0
                trig_run.value = 0
                lines[line].kill_tone()
                state = 0
                print("state 0")

        # state 3: Activating rewarder/delivering taste.
        while state == 3 and time.time() < endtime:  
            if not rew.is_crossed():
                start = time.time()
            # if rat crosses rewarder beam, deliver taste
            if rew.is_crossed() and time.time() > start + wait/10:  
                lines[line].kill_tone()
                rew_run.value = 0
                lines[line].deliver()
                print("reward delivered")
                state = 0
            if time.time() > deadline:  # if rat misses reward deadline, return to state 0
                lines[line].kill_tone()
                rew_run.value = 0
                state = 0

    trig.kill_tone()  # kill any lingering tones after task is over
    lines[line].kill_tone()

    # wait for data logging and light blinking processes 
    # to commit seppuku when session is over
    recording.join()  
    rew_flash.join()
    trig_flash.join()
    print("assay completed")


################################################################################

### SECTION 4: Menu control/"GUI", everything below runs on startup ###

# set up raspi GPIO board.
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)

## initialize objects used in task:
# initialize tastelines w/tones
# GPIO pin outputs to taste valves. Opens the valve while "1" is emitted,
tasteouts = [31, 33, 35, 37]  
# closes automatically with no voltage/ "0"
# GPIO pin outputs to intan board (for marking taste deliveries in neural data). Sends
# signal to separate device while "1" is emitted.
intanouts = [24, 26, 19, 21]  

# initialize nosepokes:
# initialize "reward" nosepoke. 
# "Rew" uses GPIO pins 38 as output for the light, and 11 as
# input for the IR sensor. For the light, 1 = On, 0 = off. 
# For the sensor, 1 = uncrossed, 0 = crossed.
rew = NosePoke(36, 11)  
# for some reason these lights come on by accident sometimes, so this turns off preemptively
rew.flash_off()  
# for some reason these lights come on by accident sometimes, so this turns off preemptively
trig.flash_off()  
