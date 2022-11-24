# pi

import time
import tempfile
from datetime import datetime

import zmq
import picamera


def main():
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind("tcp://*:5555")

  camera = picamera.PiCamera()
  camera.resolution = (2592, 1944)
  # Start a preview and let the camera warm up for 2 seconds
  camera.start_preview()
  time.sleep(2)

  while True:
    #  Wait for next request from client
    req = socket.recv()
    print("Received request: %s" % req, ' at ', datetime.now())

    camera.capture('board.jpg')
    socket.send("OK")


if __name__ == "__main__":
  main()
