"""
Code to ONLY CONTROL cue and RESPOND to nosepoke
No recording --> Leave that to Intan Board
"""

from threading import Thread, Lock
import time
import RPi.GPIO as GPIO
import datetime
import numpy as np
GPIO.setmode(GPIO.BOARD)

def trigger_action(taste_ports, laser_ports, opentime, laser_duration):
    """
    lists of ports for taste and laser
    """
    time_diff = np.float(laser_duration) - np.float(opentime)
    for port in taste_ports:  
        GPIO.output(port,1)
    for port in laser_ports:  
        GPIO.output(port,1)
    time.sleep(opentime)
    for port in taste_ports:
        GPIO.output(port,0)
    time.sleep(time_diff)
    for port in laser_ports:
        GPIO.output(port,0)

class nosepoke_thread(Thread):
    """
    Class to control cue and activate outputs 
    """
    def __init__(self, threadID, name, nosepoke_gpio, freq, 
            taste_output, laser_output, iti, cue_on, cue_off):
        """
        nosepoke_gpio :: Which port to read from
        freq :: Frequency of readings
        """
        # Inititate thread details
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name

        # Initialize board details
        GPIO.setup(nosepoke_gpio, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        # Save pin numbers
        self.nosepoke_gpio = nosepoke_gpio
        self.taste_output = taste_output
        self.laser_output = laser_output

        # Set run parameters
        self.freq = np.float(freq)
        self.cue_on = cue_on
        self.cue_off = cue_off
        self.stopped = 0
        self.thread_exit = 0

        self.iti = np.float(iti)
        self.iti_delta = datetime.timedelta(seconds = iti)
        self.latest_trigger_time = 0
        self.wait_till = datetime.datetime.now()
        self.counts = 0

    def run(self):
        print "Starting " + self.name
        self.update()
        print "Exiting " + self.name

    def update(self):
        # Keep looping indefinitely till thread is stopped
        while True:
            if not self.stopped:
                time.sleep(1/self.freq)
                #print('iter')
                if datetime.datetime.now() > self.wait_till:
                    self.cue_on()
                    #print('ON')
                temp_read = GPIO.input(self.nosepoke_gpio)
                if not temp_read: # Assuming 1 indicates poke
                    self.action_check()
            elif self.thread_exit:
                return

    def action_check(self):
        """
        Checks whether action should be allowed to pass
        """
        #print('check initiated')
        current_time = datetime.datetime.now()
        if current_time > self.wait_till:
            self.cue_off()
            #print('OFF')
            self.latest_trigger_time = current_time
            self.wait_till = current_time + self.iti_delta
            trigger_action(taste_ports, laser_ports, opentime, laser_duration)
            self.counts += 1
            print(f"Successful pokes : {self.count}, "\
                    "{current_time.strftime('%S:%M:%S')}")
            #print("ACTION COMPLETED")
        return

    def cue_status(self):
        return self.cue_on

    def set_stop(self):
        self.stopped = 1

    def release_stop(self):
        self.stopped = 0

    def exit_thread(self):
        self.thread_exit = 1

class cue_thread(Thread): 
    def __init__(self, threadID, name, cue_gpio, cue_freq):
        Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.cue_gpio = cue_gpio
        GPIO.setup(cue_gpio, GPIO.OUT) 
        GPIO.output(self.cue_gpio, 0)
        self.cue_freq = np.float(cue_freq)
        self.cue_stopped = 0
        self.thread_exit = 0

    def cue_protocol(self):
        print("Starting cue protocol")
        while True:
            time.sleep(1/self.cue_freq)
            GPIO.output(self.cue_gpio, 0)
            if not self.cue_stopped:
                time.sleep(1/self.cue_freq)
                GPIO.output(self.cue_gpio, 1)
            elif self.thread_exit:
                return

    def run(self):
        print "Starting " + self.name
        #print_time(self.name, 5, self.counter)
        self.cue_protocol()
        print "Exiting " + self.name

    def set_stop(self):
        self.cue_stopped = 1

    def release_stop(self):
        self.cue_stopped = 0

    def exit_thread(self):
        self.thread_exit = 1
        GPIO.output(self.cue_gpio, 0)

########################################
# Initiate run
########################################

if __name__ == "__main__":
    taste_out = 31
    intan_taste = 24
    laser_out = 12
    intan_laser = 8
    taste_ports = [taste_out, intan_taste]
    laser_ports = [laser_out, intan_laser]
    #opentime = 0.25

    def set_opentime(dur = None):
        global opentime
        if dur is None:
            dur = input('Please enter opentime (sec) : ')
        opentime = dur
    laser_duration = 2.5

    # Set the outports to outputs
    for i in [taste_out, intan_taste, laser_out, intan_laser]:
        GPIO.setup(i, GPIO.OUT)
        GPIO.output(i, 0)


    light = 36
    poke_sample_freq = 100
    cue_freq = 10
    beam = 11
    timeout_dur = 3

    set_opentime()

    def create_threads():
        global cue_thread1, poke_thread
        cue_thread1 = cue_thread(1, 'cue_thread', light, cue_freq)
        poke_thread = nosepoke_thread(2, 'poke_thread', beam,poke_sample_freq,1,1,timeout_dur,
                cue_thread1.release_stop, cue_thread1.set_stop)

    def start_threads():
        cue_thread1.start()
        poke_thread.start()

    def pause_threads():
        poke_thread.set_stop()
        time.sleep(2)
        cue_thread1.set_stop()
        time.sleep(2)
        cue_thread1.set_stop()

    def unpause_threads():
        poke_thread.release_stop()
        cue_thread1.release_stop()

    def exit_threads():
        pause_threads()
        poke_thread.exit_thread()
        cue_thread1.exit_thread()

