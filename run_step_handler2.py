import logging
import argparse
import time
import uuid
import os

import cv2
import numpy as np
import zmq
import hydra
from omegaconf import OmegaConf, DictConfig

from threes_ai.threes.hanlder import StepHandler

handler = None


def on_img_received(img, flags):
  if not os.path.exists(flags.save_img_path):
    os.makedirs(flags.save_img_path)

  # Save img to a tmp folder for training models.
  img_path = os.path.join(flags.save_img_path, str(uuid.uuid4()) + ".jpg")
  cv2.imwrite(img_path, img)
  logging.info("Image saved: %s", img_path)

  global handler
  if handler is None:
    handler = StepHandler(flags)

  handler.execute(img_path)


@hydra.main(config_path="conf", config_name="step_handler_config")
def main(flags: DictConfig):

  #  Socket to talk to server
  logging.info("Connecting to pi image server…")
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://192.168.31.207:5555")

  while True:
    logging.info("request pi...")
    socket.send(b'get')

    # Get the reply.
    buf = socket.recv_pyobj()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    on_img_received(img, flags)
    input("wait for moving...")


if __name__ == "__main__":
  main()
