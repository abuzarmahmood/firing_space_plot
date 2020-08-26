# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:15:36 2019

@author: dsvedberg
"""
import time
import multiprocessing
import RPi.GPIO as GPIO
import os
import numpy as np
import datetime

class nosePoke:
        def __init__(self,light, beam, name):
                self.light = light
                self.beam = beam
                self.name = name
                GPIO.setup(self.light,GPIO.OUT)
                GPIO.setup(self.beam,GPIO.IN)

        def shutdown(self):
                print("blink shutdown")
                self.exit.set()
                
        def flashOn(self):
                GPIO.output(self.light,1)
                
        def flashOff(self):
                GPIO.output(self.light,0)
                
        def flash(self, Hz):
                while True:
                        self.flashOn()
                        time.sleep(2/Hz)
                        self.flashOff()
                        time.sleep(2/Hz)
                        
        def isCrossed(self):
                if GPIO.input(self.beam) == 0: return True
                else: return False
                
        def sensorTrip(self,endtime,wait,ret_val):
                ret_val[1] = 0
                start = time.time()
                while time.time() < endtime:
                        if GPIO.input(self.beam) == 1:
                                start = time.time()
                        if GPIO.input(self.beam) == 0 and time.time()-start > wait:
                                print("sensor_tripped")
                                ret_val[1] = 1
                                return
                                
        def keepOut(self,endtime,wait):
                start = time.time()
                while True and time.time() < endtime:         
                        if self.isCrossed():
                                start = time.time()
                        elif time.time()-start > wait:
                                break
                        
        def kill(self):
                GPIO.output(self.light,0)

class tasteLine:
        def __init__(self,valve,intanOut):
                self.valve = valve
                self.intanOut = intanOut
                self.opentime = 0.05
                GPIO.setup(self.valve,GPIO.OUT)
                GPIO.setup(self.intanOut,GPIO.OUT)
                
        def clearout(self):
                dur = input("enter a clearout time to start clearout, or enter '0' to cancel: ")
                if dur == 0:
                        print("clearout canceled")
                        return
                GPIO.output(self.valve, 1)
                time.sleep(dur)
                GPIO.output(self.valve, 0)
                print('Tastant line clearing complete.')
                
        def calibrate(self):
                opentime = input("enter an opentime (like 0.05) to start calibration: ")
                
                while True:
                        # Open ports  
                        for rep in range(5):
                                GPIO.output(self.valve, 1)
                                time.sleep(opentime)
                                GPIO.output(self.valve, 0)
                                time.sleep(3)
        
                        ans = raw_input('keep this calibration? (y/n)')
                        if ans == 'y':
                                self.opentime = opentime
                                print("opentime saved")
                                break
                        else:
                                opentime = input('enter new opentime:')
        def deliver(self):
                GPIO.output(self.valve, 1)
                GPIO.output(self.intanOut, 1)
                time.sleep(self.opentime)
                GPIO.output(self.valve, 0)
                GPIO.output(self.intanOut, 0)
        
        def kill(self):
                GPIO.output(self.valve, 0)
                GPIO.output(self.intanOut, 0)
                
        def isOpen(self):
                if GPIO.input(self.valve): return True
                else: return False
                
def play1000hz():
        os.system('omxplayer --loop 1000hz_sine.mp3 &')
        
def saveData(dels,waits):
        while True:
                a = raw_input("assay finished, would you like to save data? (y/n): ")
                try:
                        if a == 'n':
                                print("latency data not saved")
                                break
                        else:
                                anID = raw_input("enter the animal ID: ")
                                now = datetime.datetime.now()
                                d = now.strftime("%m%d%y_%Hh%Mm")
                                try: 
                                        np.save(anID+"_"+d+"_"+"lt", dels)
                                        np.save(anID+"_"+d+"_"+"wt", waits)
                                        print("file saved")
                                        break
                                except:
                                        print("input error try again")
                except:
                        print("input error try again")

def saveDoubleData(dels,acts,poke1Times,poke2Times):
        while True:
                a = raw_input("assay finished, would you like to save data? (y/n): ")
                try:
                        if a == 'n':
                                print("latency data not saved")
                                break
                        else:
                                try:
                                        anID = raw_input("enter the animal ID: ")
                                        now = datetime.datetime.now()
                                        d = now.strftime("%m%d%y_%Hh%Mm")
                                        np.save(anID+"_"+d+"_"+"lt", dels)
                                        np.save(anID+"_"+d+"_"+"at", acts)
                                        np.save(anID+"_"+d+"_"+"poke1Times", poke1Times)
                                        np.save(anID+"_"+d+"_"+"poke2Times", poke2Times)
                                        print("file saved")
                                except:
                                        print("input error try again")
                                break
                except:
                        print("input error try again")
                print("exit file save")
                        
def pokeHab():
        runtime = input("enter runtime in minutes: ")
        poke1 = nosePoke(36,11)
        dels  = []
        waittimes = []
        starttime = time.time()
        endtime = starttime+runtime*60
        loop = 0
        wait = 0.1
        while True:
                if loop > 19:
                        wait = 2
                elif loop > 14:
                        wait = 1.5
                elif loop > 9:
                        wait = 1
                elif loop > 4:
                        wait = 0.5
                manager = multiprocessing.Manager()
                ret_val = manager.dict()
                sensing = multiprocessing.Process(target = poke1.sensorTrip, args = (endtime,wait,ret_val,))
                audio = multiprocessing.Process(target = play1000hz, args = ())
                flashing = multiprocessing.Process(target = poke1.flash, args = (3.9,))
                audio.start()
                flashing.start()
                sensing.start()
                sensing.join()
                dels = np.append(dels,time.time()-starttime)
                waittimes = np.append(waittimes,wait)
                lines[0].deliver()
                audio.terminate()
                flashing.terminate()
                poke1.flashOff()
                os.system('killall omxplayer.bin')
                poke1.keepOut(endtime,wait)
                if time.time() > endtime:
                        flashing.terminate()
                        sensing.terminate()
                        audio.terminate()
                        os.system('killall omxplayer.bin')
                        break
                loop = loop+1
        saveData(dels,waittimes)
        
def pokeHab2():
        runtime = input("enter runtime in minutes: ")
        poke1 = nosePoke(36,11)
        dels  = []
        waittimes = []
        starttime = time.time()
        endtime = starttime+runtime*60
        loop = 0
        wait = 1
        while True:
                manager = multiprocessing.Manager()
                ret_val = manager.dict()
                sensing = multiprocessing.Process(target = poke1.sensorTrip, args = (endtime,wait,ret_val,))
                audio = multiprocessing.Process(target = play1000hz, args = ())
                flashing = multiprocessing.Process(target = poke1.flash, args = (3.9,))
                audio.start()
                flashing.start()
                sensing.start()
                sensing.join()
                dels = np.append(dels,time.time()-starttime)
                waittimes = np.append(waittimes,wait)
                lines[0].deliver()
                audio.terminate()
                flashing.terminate()
                poke1.flashOff()
                os.system('killall omxplayer.bin')
                poke1.keepOut(endtime,wait)
                if time.time() > endtime:
                        flashing.terminate()
                        sensing.terminate()
                        audio.terminate()
                        os.system('killall omxplayer.bin')
                        break
                loop = loop+1
        saveData(dels,waittimes)
        
def doublePokeHab():
        runtime = input("enter runtime in minutes: ")
        anID = raw_input("enter animal ID: ")
        poke1 = nosePoke(36,11,'poke1')
        poke2 = nosePoke(38,13,'poke2')
        starttime = time.time()
        endtime = starttime+runtime*60
        loop = 0
        wait = 0.5
        crosstime = 10
        manager = multiprocessing.Manager()
        recording = multiprocessing.Process(target = record, args = (poke1, poke2, lines[0], lines[1],starttime,endtime,anID,))
        recording.start()
        
        while True:
                ret_val = manager.dict()
                
                sensing2 = multiprocessing.Process(target = poke2.sensorTrip, args = (endtime,wait,ret_val,))
                flashing2 = multiprocessing.Process(target = poke2.flash, args = (3.9,))
                
                audio = multiprocessing.Process(target = play1000hz, args = ())
                flashing = multiprocessing.Process(target = poke1.flash, args = (3.9,))
                flashing2.start()
                sensing2.start()
                sensing2.join()
                flashing2.terminate()
                poke2.flashOff()
                
                print(ret_val[1])
                
                if ret_val[1] == 1:
                        audio.start()
                        flashing.start()
                        print("activated!")
                        lines[1].deliver()
                        ret_val = manager.dict()
                        deadline = time.time()+crosstime
                        sensing = multiprocessing.Process(target = poke1.sensorTrip, args = (deadline,wait,ret_val,))
                        sensing.start()
                        sensing.join()
                
                        if time.time() < endtime and ret_val[1] == 1:
                                lines[0].deliver()
                                loop = loop+1
                                print("reward delivered!")
                        audio.terminate()
                        flashing.terminate()
                        poke1.flashOff()
                        os.system('killall omxplayer.bin')
                        poke1.keepOut(endtime,wait)
                        
                if time.time() > endtime:
                        try:
                                recording.terminate()
                                flashing2.terminate()
                                flashing.terminate()
                                sensing2.terminate()
                                sensing.terminate()
                                audio.terminate()
                                os.system('killall omxplayer.bin')
                                break
                        except:
                                break
        recording.join()
        print("assay completed")
        
def record(poke1,poke2,line1,line2,starttime,endtime,anID):
        now = datetime.datetime.now()
        d = now.strftime("%m%d%y_%Hh%Mm")
        localpath = os.getcwd()
        filepath = localpath+"/"+anID+"_"+d+".csv"
        file = open(filepath,"a")
        if os.stat(filepath).st_size == 0:
                file.write("Time,Poke1,Poke2,Line1,Line2\n")
        while time.time() < endtime:
                t = round(time.time()-starttime,2)
                file.write(str(t)+","+str(poke1.isCrossed())+","+str(poke2.isCrossed())+","+str(line1.isOpen())+","+str(line2.isOpen())+"\n")
                file.flush()
                time.sleep(0.1)
        file.close()
        
def killAll():
        GPIO.cleanup()
        os.system("killall omxplayer.bin")
        
def main_menu():       ## Your menu design here
        options = ["clearout a line","calibrate a line", "pokeHab1","pokeHab2", "doublePokeHab","kill all","exit"]
        print(30 * "-" , "MENU" , 30 * "-")
        for idx,item in enumerate(options):
                print(str(idx+1)+". "+item)
        print(67 * "-")
        choice = input("Enter your choice [1-7]: ")
        return choice
        
def clearout_menu():
        while True:
                for x in range(1,5):
                        print(str(x)+". clearout line "+str(x))
                print("5. main menu")
                line = input("enter your choice")
                if line in range(1,6):
                        return(line-1)
                else:
                        print("enter a valid menu option")
                
                      
def calibration_menu():
        while True:
                for x in range(1,5):
                        print(str(x)+". calibrate line "+str(x))
                print("5. main menu")
                line = input("enter your choice")
                if line in range(1,6):
                        return(line-1)
                else:
                        print("enter a valid menu option")

##main##
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)  
tasteouts = [31,33,35,37]
intanouts = [24,26,19,21]

lines = [tasteLine(tasteouts[i],intanouts[i]) for i in range(4)]

while True:     
        ## While loop which will keep going until loop = False
        choice = main_menu()    ## Displays menu
        try:
                if choice ==1:
                        while True:
                                line = clearout_menu()
                                if line in range(4):
                                        lines[line].clearout()
                                elif line == 4:
                                        break
                elif choice==2:     
                        while True:
                                line = calibration_menu()
                                if line in range(4):
                                        lines[line].calibrate()
                                elif line == 4:
                                        break               
                elif choice==3:
                        print("starting pokeHab")
                        pokeHab()
                elif choice==4:
                        print("starting pokeHab2")
                        pokeHab2()
                elif choice==5:
                        print("starting doublePokeHab")
                        doublePokeHab()
                elif choice==6:
                        print("killing all")
                        killAll()
                elif choice==7:
                        print("program exit")
                        break
                        
        except ValueError:
                print("please enter a number: ")
                
