import argparse
import time
import uuid
import os

import cv2
import numpy as np
import zmq


def on_img_received(img, args):
  img_path = os.path.join(args.output, str(uuid.uuid4()) + ".jpg")
  cv2.imwrite(img_path, img)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      help="Dest folder for save images.",
                      required=True)
  args = parser.parse_args()

  #  Socket to talk to server
  print("Connecting to pi image server…")
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://192.168.31.207:5555")

  while True:
    print("request img...")
    socket.send(b'get')

    # Get the reply.
    buf = socket.recv_pyobj()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    print("Received image ", img.shape)

    on_img_received(img, args)

    # Do some 'work'
    time.sleep(2)


if __name__ == "__main__":
  main()
