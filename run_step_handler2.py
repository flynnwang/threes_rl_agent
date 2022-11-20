import logging
import argparse
import time
import uuid
import os

import cv2
import numpy as np
import zmq

from threes_ai.threes.hanlder import StepHandler


def on_img_received(img, flags):
  # Save img to a tmp folder for training models.
  img_path = os.path.join(args.save_img_path, str(uuid.uuid4()) + ".jpg")
  cv2.imwrite(img_path, img)

  handler = StepHandler(flags)
  handler.execute(img_path)


@hydra.main(config_path="conf", config_name="step_handler_config")
def main():

  #  Socket to talk to server
  logging.info("Connecting to pi image server…")
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://192.168.31.207:5555")

  while True:
    input("request...pi...")
    socket.send(b'get')

    # Get the reply.
    buf = socket.recv_pyobj()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    logging.info("Image received: ", img.shape)

    on_img_received(img, args)


if __name__ == "__main__":
  main()
