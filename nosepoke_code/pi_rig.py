'''
pi_rig contrains basic functions for using the raspberry pi behavior and electrophysiology rig in the Katz Lab

These functions can be used directly via ipython in a terminal window or called by other codes
'''

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

# To empty taste lines
def clearout(outports = [31, 33, 35, 37], dur = 5):

        # Setup pi board GPIO ports
	GPIO.setmode(GPIO.BOARD)
	for i in outports:
		GPIO.setup(i, GPIO.OUT)

	for i in outports:
		GPIO.output(i, 1)
	time.sleep(dur)
	for i in outports:
		GPIO.output(i, 0)

	print('Tastant line clearing complete.')
	
	
# To calibrate taste lines
def calibrate(outports = [31, 33, 35, 37], opentime = 0.015, repeats = 5):

        # Setup pi board GPIO ports
	GPIO.setmode(GPIO.BOARD)
	for i in outports:
		GPIO.setup(i, GPIO.OUT)

        # Open ports  
	for rep in range(repeats):
		for i in outports:
			GPIO.output(i, 1)
		time.sleep(opentime)
		for i in outports:
			GPIO.output(i, 0)
		time.sleep(3)

	print('Calibration procedure complete.')
	
	
# Passive H2O deliveries
def passive(outports = [31, 33, 35, 37], intaninputs = [24, 26, 19, 21], opentimes = [0.01], itimin = 10, itimax = 30, trials = 150):


        # Setup pi board GPIO ports
	GPIO.setmode(GPIO.BOARD)
	for i in outports:
		GPIO.setup(i, GPIO.OUT)
	for i in intaninputs:
		GPIO.setup(i, GPIO.OUT)

        # Set and radomize trial order
        tot_trials = len(outports) * trials
        count = 0
        trial_array = trials * range(len(outports))
        random.shuffle(trial_array)

	time.sleep(15)
	
	# Loop through trials
	for i in trial_array:
		GPIO.output(outports[i], 1)
		GPIO.output(intaninputs[i], 1)
		time.sleep(opentimes[i])
		GPIO.output(outports[i], 0)
		GPIO.output(intaninputs[i], 0)
		count += 1
		iti = random.randint(itimin, itimax)
		print('Trial '+str(count)+' of '+str(tot_trials)+' completed. ITI = '+str(iti)+' sec.')
		time.sleep(iti)

	print('Passive deliveries completed')
	
# Passive H2O deliveries
def passive_cue(outports = [31, 33, 35, 37], intaninputs = [24, 26, 19, 21], opentimes = [0.01], itimin = 10, itimax = 30, trials = 150):


        # Setup pi board GPIO ports
	GPIO.setmode(GPIO.BOARD)
	for i in outports:
		GPIO.setup(i, GPIO.OUT)
	for i in intaninputs:
		GPIO.setup(i, GPIO.OUT)
	GPIO.setup(18, GPIO.OUT)

        # Set and radomize trial order
        tot_trials = len(outports) * trials
        count = 0
        trial_array = trials * range(len(outports))
        random.shuffle(trial_array)

	time.sleep(15)
	
	# Loop through trials
	for i in trial_array:
		GPIO.output(18, 1)
		time.sleep(1)
		GPIO.output(18, 0)
		time.sleep(1)
		GPIO.output(outports[i], 1)
		GPIO.output(intaninputs[i], 1)
		time.sleep(opentimes[i])
		GPIO.output(outports[i], 0)
		GPIO.output(intaninputs[i], 0)
		count += 1
		iti = random.randint(itimin, itimax)
		print('Trial '+str(count)+' of '+str(tot_trials)+' completed. ITI = '+str(iti)+' sec.')
		time.sleep(iti)

	print('Passive deliveries completed')

# Passive deliveries with video recordings
def passive_with_video(outports = [31, 33, 35, 37], intan_inports = [24, 26, 19, 21], tastes = ['water', 'sucrose', 'NaCl', 'quinine'], opentimes = [0.015, 0.015, 0.015, 0.015], iti = 15, repeats = 30):

	# Set the outports to outputs
	GPIO.setmode(GPIO.BOARD)
	for i in outports:
		GPIO.setup(i, GPIO.OUT)

	# Set the input lines for Intan to outputs
	for i in intan_inports:
		GPIO.setup(i, GPIO.OUT)
		GPIO.output(i, 0)


	# Define the port for the video cue light, and set it as output
	video_cue = 16
	GPIO.setup(video_cue, GPIO.OUT)

	# Make an ordered array of the number of tastes (length of outports)
	taste_array = []
	for i in range(len(outports)*repeats):
		taste_array.append(int(i%len(outports)))

	# Randomize the array of tastes, and print it
	np.random.shuffle(taste_array)
	print "Chosen sequence of tastes:" + '\n' + str(taste_array)

	# Ask the user for the directory to save the video files in	
	directory = easygui.diropenbox(msg = 'Select the directory to save the videos from this experiment', title = 'Select directory')
	# Change to that directory
	os.chdir(directory)

	# A 10 sec wait before things start
	time.sleep(10)

	# Deliver the tastes according to the order in taste_array
	trials = [1 for i in range(len(outports))]
	for taste in taste_array:
		# Make filename, and start the video in a separate process
		process = Popen('sudo streamer -q -c /dev/video0 -s 1280x720 -f jpeg -t 180 -r 30 -j 75 -w 0 -o ' + tastes[taste] + '_trial_' + str(trials[taste]) + '.avi', shell = True, stdout = None, stdin = None, stderr = None, close_fds = True)

		# Wait for 2 sec, before delivering tastes
		time.sleep(2)

		# Switch on the cue light
		GPIO.output(video_cue, 1)

		# Deliver the taste, and send outputs to Intan
		GPIO.output(outports[taste], 1)
		GPIO.output(intan_inports[taste], 1)
		time.sleep(opentimes[taste])	
		GPIO.output(outports[taste], 0)
		GPIO.output(intan_inports[taste], 0)

		# Switch the light off after 50 ms
		time.sleep(0.050)
		GPIO.output(video_cue, 0)

		# Increment the trial counter for the taste by 1
		trials[taste] += 1

		# Print number of trials completed
		print "Trial " + str(np.sum(trials) - len(outports)) + " of " + str(len(taste_array)) + " completed."

		# Wait for the iti before delivering next taste
		time.sleep(iti)


# Basic nose poking procedure to train poking for discrimination 2-AFC task
def basic_np(outport = 31, opentime = 0.012, iti = [.4, 1, 2], trials = 200, outtime = 0):

	intaninput = 8
	trial = 1
	inport = 13
	pokelight = 38
	houselight = 18
	lights = 0
	maxtime = 60

        # Setup pi board GPIO ports 
        GPIO.setmode(GPIO.BOARD)
	GPIO.setup(pokelight, GPIO.OUT)
	GPIO.setup(houselight, GPIO.OUT)
	GPIO.setup(inport, GPIO.IN)
	GPIO.setup(outport, GPIO.OUT)
	GPIO.setup(intaninput, GPIO.OUT)
	
	time.sleep(15)
	starttime = time.time()

	while trial <= trials:

                # Timer to stop experiment if over 60 mins
		curtime = time.time()
		elapsedtime = round((curtime - starttime)/60, 2)
		if elapsedtime > maxtime:
			GPIO.output(pokelight, 0)
			GPIO.output(houselight, 0)
			break

		if lights == 0:
			GPIO.output(pokelight, 1)
			GPIO.output(houselight, 1)
			lights = 1

                # Check for pokes
		if GPIO.input(inport) == 0:
			poketime = time.time()
			curtime = poketime

                        # Make rat remove nose from nose poke to receive reward
			while (curtime - poketime) <= outtime:
				if GPIO.input(inport) == 0:
					poketime = time.time()
				curtime = time.time()

                        # Taste delivery and switch off lights
			GPIO.output(outport, 1)
			GPIO.output(intaninput, 1)
			time.sleep(opentime)
			GPIO.output(outport, 0)
			GPIO.output(intaninput, 1)
			GPIO.output(pokelight, 0)
			GPIO.output(houselight, 0)
			print('Trial '+str(trial)+' of '+str(trials)+' completed.')
			trial += 1
			lights = 0

                        # Calculate and execute ITI delay.  Pokes during ITI reset ITI timer.
			if trial <= trials/2:
				delay = floor((random.random()*(iti[1]-iti[0]))*100)/100+iti[0]
			else:
				delay = floor((random.random()*(iti[2]-iti[0]))*100)/100+iti[0]
	
			poketime = time.time()
			curtime = poketime

			while (curtime - poketime) <= delay:
				if GPIO.input(inport) == 0:
					poketime = time.time()
				curtime = time.time()
		
	print('Basic nose poking has been completed.')

# Passive H2O deliveries
def affective(intaninputs = [24], tim_dur = 1200):


        # Setup pi board GPIO ports
	GPIO.setmode(GPIO.BOARD)
	for i in intaninputs:
		GPIO.setup(i, GPIO.OUT)

	
	# Loop through trials
	
	GPIO.output(intaninputs[0], 1)
	time.sleep(0.1)
	GPIO.output(intaninputs[0], 0)

	time.sleep(tim_dur)
	
	GPIO.output(intaninputs[0], 1)
	time.sleep(0.1)
	GPIO.output(intaninputs[0], 0)


	print('Test completed')
	
	
# Clear all pi board GPIO settings
def clearall():

	# Pi ports to be cleared
	outports = [31, 33, 35, 37]
	inports = [11, 13, 15]
	pokelights = [36, 38, 40]
	houselight = 18
	lasers = [12, 22, 16]
	intan = [8, 10, 24, 26, 19, 21]
	
	# Set all ports to default/low state
	for i in intan:
		GPIO.setup(i, GPIO.OUT)
		GPIO.output(i, 0)
	
	for i in outports:
		GPIO.setup(i, GPIO.OUT)
		GPIO.output(i, 0)
		
	for i in inports:
		GPIO.setup(i, GPIO.IN, GPIO.PUD_UP)
		
	for i in pokelights:
		GPIO.setup(i, GPIO.OUT)
		GPIO.output(i, 0)
		
	for i in lasers:
		GPIO.setup(i, GPIO.OUT)
		GPIO.output(i, 0)
		
	GPIO.setup(houselight, GPIO.OUT)
	GPIO.output(houselight, 0)

