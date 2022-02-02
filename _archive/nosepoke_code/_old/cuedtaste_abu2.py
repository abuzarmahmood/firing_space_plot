from threading import Thread
import time
#import RPi.GPIO as GPIO
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

class nosepoke_io:
    """
    Class to read/write from nosepoke in threads 
    """
    def __init__(self, nosepoke_gpio, freq, outfile):
        """
        nosepoke_gpio :: Which port to read from
        freq :: Frequency of readings
        """
        # Initialize board details
        #GPIO.setmode(GPIO.BOARD)
        #GPIO.setup(nosepoke_gpio, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        self.nosepoke_gpio = nosepoke_gpio
        self.freq = freq
        self.outfile = outfile
        self.out_connect = open(self.outfile, "a")
        self.buffer = []
        self.timelist = []
        self.poke_bool = 0
        self.stopped = 0

    def update(self):
        # Keep looping indefinitely till thread is stopped
        while True:
            time.sleep(1/self.freq)
            if self.stopped:
                return
            #temp_read = GPIO.input(self.nosepoke_gpio)
            temp_read = np.random.choice([0,1])
            self.poke_bool = temp_read # Assuming poke generates a 1
            self.buffer.append(temp_read)
            self.timelist.append(datetime.datetime.now())

    def write_out(self):
        # Keep looping indefinitely and writeout whenever a new datapoint is collected
        while True:
            time.sleep(1/self.freq) # 2x read freq
            if self.stopped:
                return
            if len(self.buffer): 
                self.out_connect.write(f"{self.timelist.pop(0)},{self.buffer.pop(0)}\n")
                self.out_connect.flush()

    def start_read(self):
        # Start thread to read from gpio
        t = Thread(target = self.update, name = 'read_thread', args = ())
        t.daemon = True
        t.start()
        return self

    def start_write(self):
        # Start thread to write from buffer 
        t = Thread(target = self.write_out, name = 'write_thread', args = ())
        t.daemon = True
        t.start()
        return self

    def return_read(self):
        return self.buffer[-1]

    def stop_all(self):
        self.stopped = True
        self.out_connect.close()

class trigger_handler:
    """
    Class to respond to boolean nosepoke signal
    Runs thread to continuously check on trigger signal
    Once signal is detected, checks are executed to make sure output timing is appropriate
    If yes, output is generated
    """
    def __init__(self, trigger_signal, taste_output, laser_output, freq, iti):
        """
        iti :: wait time before next trigger, in seconds
        """
        self.trigger_signal = trigger_signal
        self.taste_output = taste_output
        self.laser_output = laser_output
        self.freq = freq
        self.iti = datetime.timedelta(seconds = iti)
        self.stopped = 0
        self.latest_trigger_time = 0
        self.wait_till = datetime.datetime.now()
        self.cue_freq = 2
        self.cue_on = 1

    def action_check(self):
        """
        Checks whether action should be allowed to pass
        """
        current_time = datetime.datetime.now()
        print("Check initiated")
        if current_time > self.wait_till:
            self.cue_on = 0
            self.latest_trigger_time = current_time
            self.wait_till = current_time + self.iti
            print("ACTION COMPLETED")
        return

    def initiate_check(self):
        while True:
            if not self.stopped:
                time.sleep(0.1/self.freq) # 2x read freq
                if self.trigger_signal():
                    self.action_check()
        return

    def cue_protocol(self):
        while True:
            if not self.stopped:
                if (datetime.datetime.now() > self.wait_till) or self.cue_on:
                    print("Flash")
                    print("|")
                    time.sleep(0.5/self.cue_freq)
                    print("--")

    def start_checks(self):
        # Start thread to write from buffer 
        t = Thread(target = self.initiate_check(), name = 'check_thread', args = ())
        t.daemon = True
        t.start()
        return self

    def start_cue(self):
        # Start thread to write from buffer 
        t = Thread(target = self.cue_protocol(), name = 'check_thread', args = ())
        t.daemon = True
        t.start()
        return self

freq = 10
test_poke_io = nosepoke_io(1,freq,'test_out.txt')
test_poke_io.start_read()
test_poke_io.start_write()

test_trigger = trigger_handler(test_poke_io.return_read, 1,1, freq, 5)
test_trigger.start_checks()
test_trigger.start_cue()
