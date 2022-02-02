"""
Code to ONLY CONTROL cue and RESPOND to nosepoke
No recording --> Leave that to Intan Board
"""

from threading import Thread, Lock
import time
import RPi.GPIO as GPIO
import datetime
import numpy as np


class nosepoke_task:
    """
    - Class to run nosepoke task
    - Stores main variables and performs appropriate setup
    - Will allow delivery of tastant and control of laser dependent on nosepoke parameters
    - No need to involve INTAN board since laser is either on or off for all trials
        and nosepoke parameters will stay the same
    """

class nosepoke_trigger:
    """
    Class to control cue and activate outputs 
    """
    def __init__(self, nosepoke_gpio, cue_gpio, freq, 
            taste_output, laser_output, iti):
        """
        nosepoke_gpio :: Which port to read from
        freq :: Frequency of readings
        """
        # Initialize board details
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(nosepoke_gpio, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(cue_gpio, GPIO.OUT) 

        self.nosepoke_gpio = nosepoke_gpio
        self.cue_gpio = cue_gpio
        self.freq = np.float(freq)
        self.poke_bool = 0
        self.stopped = 0

        self.taste_output = taste_output
        self.laser_output = laser_output
        self.freq = freq
        self.iti_delta = datetime.timedelta(seconds = iti)
        self.latest_trigger_time = 0
        self.wait_till = datetime.datetime.now()
        self.iti = iti
        self.cue_freq = np.float(2)
        self.cue_on = 1

    def update(self):
        # Keep looping indefinitely till thread is stopped
        while True:
            time.sleep(1/self.freq)
            if self.stopped:
                return
            temp_read = GPIO.input(self.nosepoke_gpio)
            #temp_read = np.random.choice([0,1], p = [0.9,0.1])
            #print(temp_read)
            if not temp_read: # Assuming 1 indicates poke
                self.action_check()

    def action_check(self):
        """
        Checks whether action should be allowed to pass
        """
        current_time = datetime.datetime.now()
        #print("Check initiated")
        #if current_time > self.wait_till:
        self.cue_on = 0
        #self.latest_trigger_time = current_time
        #self.wait_till = current_time + self.iti
        print("ACTION COMPLETED")
        time.sleep(self.iti)
        self.cue_on = 1
        return

    def cue_protocol(self):
        while True:
            time.sleep(1/self.cue_freq)
            GPIO.output(self.cue_gpio, 0)
            if not self.stopped:
                time.sleep(0.5/self.cue_freq)
                GPIO.output(self.cue_gpio, 1)

    def start_update(self):
        # Start thread to write from buffer 
        t = Thread(target = self.update(), name = 'check_thread', args = ())
        t.daemon = True
        t.start()
        return self

    def start_cue(self):
        # Start thread to write from buffer 
        t = Thread(target = self.cue_protocol(), name = 'cue_thread', args = ())
        t.daemon = True
        t.start()
        return self

    def stop_all(self):
        self.stopped = True
        self.out_connect.close()

freq = 100
light = 36
beam = 11
test_poke_io = nosepoke_trigger(beam,light,freq,1,1,10)
test_poke_io.start_update()
test_poke_io.start_cue()
