#!/usr/bin/env python3

import threading 
from threading import Thread
import cv2
import numpy as np
import base64
import asyncio
from multiprocessing import Queue

#Creating global variables for Extraction
exLock = threading.Lock()
extractFSem = asyncio.Semaphore(10)
extractESem = asyncio.Semaphore(10)
extractQueue = Queue()

# Creating global variables for conversion
conLock = threading.Lock()
conFSem = asyncio.Semaphore(10)
conESem = asyncio.Semaphore(10)
conQueue = Queue()

# Global filename
filename = 'clip.mp4'
frameCount = 0

class Producer(threading.Thread):
    # Producer class takes in a shared queue, two semaphores for full and empty and a lock
    # Set up the definition
    def __init__(self, sharedque, fullsem, emptysem, qlock):
        threading.Thread.__init__(self)
        self.sharedque =sharedque
        self.fullsem = fullsem
        self.emptysem = emptysem
        self.qlock = qlock
        print("Initialized")

    # Method to work with the queue it checks it is able to be used, and then locks the queue adds an item and changes the full semaphore
    def produce(self, item):
        self.emptysem.acquire()
        self.qlock.acquire()
        self.sharedque.put(item)
        self.qlock.release()
        self.fullsem.release()

    #Debug produce
    def dproduce(self,item):
        self.emptysem.acquire()
        print("Checked empty semaphore")
        self.qlock.acquire()
        print("Lock acquired")
        self.sharedque.put(item)
        print("Added item")
        self.qlock.release()
        print("Lock relesed")
        self.fullsem.release()
        print("Full semaphore released")
        print("Done with item")

class Consumer(threading.Thread):
    # Producer class takes in a shared queue, two semaphores for full and empty and a lock
    # Set up the definition
    def __init__(self, sharedque, fullsem, emptysem, qlock):
        threading.Thread.__init__(self)
        self.sharedque =sharedque
        self.fullsem = fullsem
        self.emptysem = emptysem
        self.qlock = qlock
        print("Initialized")

    # Method to work with the queue it checks it is able to be used, and then locks the queue gets an item and changes the full semaphore
    def consume(self):
        self.fullsem.acquire()
        self.qlock.acquire()
        item=self.sharedque.get()
        self.qlock.release()
        self.emptysem.release()
        return item

    #Debug consume
    def dconsume(self):
        self.fullsem.acquire()
        print("Checked full semaphore")
        self.qlock.acquire()
        print("Lock acquired")
        item = self.sharedque.get()
        print("Obtained item")
        self.qlock.release()
        print("Lock relesed")
        self.emptysem.release()
        print("Empty semaphore released")
        print("Done with item")
        return item
    
    
ExtPro = Producer(extractQueue, extractFSem, extractESem, exLock)
ExtCon = Consumer(extractQueue, extractFSem, extractESem, exLock)
ConPro = Producer(conQueue, conFSem, conESem, conLock)
ConCon = Consumer(conQueue, conFSem, conESem, conLock)

# Method for extracting the frames
def extractFrames(fileName, proBuf):
    # Initialize frame count
    count = 0

    # Open video file
    vidcap = cv2.VideoCapture(fileName)
    
    # Read first image
    success, image = vidcap.read()

    print("Reading frame {} {} ".format(count, success))
    while success:
        # get a jpg encoded frame
        success, jpgImage =cv2.imencode('.jpg', image)

        # encode the frame as base 64 to make debugging easier
        jpgAsText = base64.b64encode(jpgImage)

        # add the frame to the buffer
        proBuf.produce(jpgAsText)
        # debug use buffer
        #proBuf.put(jpgAsText)

        success, image = vidcap.read()
        print('Reading frame {} {}'.format(count, success))
        count+=1
        
    print("Frame extraction complete")

# Method for converting frames to grey
def convertFrames(conBuf, convPro):
    # Initialize frame count
    count = 0

    while (count < 739):
        # Get the frame
        frameAsText = conBuf.consume()

        # Decode the frame
        jpgRawImage = base64.b64decode(frameAsText)

        # Convert the raw frame to a numpy array
        jpgImage = np.asarray(bytearray(jpgRawImage), dtype=np.uint8)
        
        # get a jpg encoded frame
        img = cv2.imdecode( jpgImage, cv2.IMREAD_COLOR)

        #Convert frame
        print("Converting frame {}".format(count))
        grayscaleFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Reencode frame
        #convJpgImage = cv2.imencode('.jpg', grayscaleFrame)

        # Encode the frame as base 64
        convJpgAsText = base64.b64encode(grayscaleFrame)
        
        # Add to second buffer
        convPro.produce(convJpgAsText)
        print("Adding converted frame {}".format(count))
        

        count +=1

    print("Done converting")

# Displaying method
def displayFrames(convCon):
    # Initialize frame count
    count = 0

    while(count < 739):
        # Get the frame
        frameAsText = convCon.consume()

        # Decode the frame
        jpgRawImage = base64.b64decode(frameAsText)
        
        # Convert the raw frame toa numpy array
        jpgImage = np.asarray(bytearray(jpgRawImage), dtype=np.uint8)

        # Get a jpg encoded frame
        img = cv2.imdecode( jpgImage, cv2.IMREAD_UNCHANGED)

        print("Displaying frame {}".format(count))

        # Display the image in a window called "video" and wait 42ms
        # before displaying teh next frame
        cv2.imshow("Video", img)
        if cv2.waitKey(42) and 0xFF == ord("q"):
            break

        count += 1

    print("Finished displaying all frames")
    cv2.destroyAllWindows()




threads = []
thread0 = threading.Thread(target=extractFrames(filename, ExtPro))
thread1 = threading.Thread(target=convertFrames(ExtCon, ConPro))
thread2 = threading.Thread(target=displayFrames(ConCon))

threads.append(thread0)
threads.append(thread1)
threads.append(thread2)

for thread in threads:
    thread.start()
