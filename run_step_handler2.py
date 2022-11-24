import logging
import argparse
import time
import uuid
import os
import shutil

import requests
import cv2
import numpy as np
import zmq
import hydra
from omegaconf import OmegaConf, DictConfig

from threes_ai.threes.hanlder import StepHandler

handler = None

IMAGE_URL = "http://192.168.31.207:8000/board.jpg"


def download_img(img_path):
  r = requests.get(IMAGE_URL, stream=True)
  if r.status_code == 200:
    with open(img_path, 'wb') as f:
      r.raw.decode_content = True
      shutil.copyfileobj(r.raw, f)
    logging.info("Image downloaded: %s", img_path)
    return True
  logging.error("Failed to download image file!")
  return False


def on_img_received(flags):
  if not os.path.exists(flags.save_img_path):
    os.makedirs(flags.save_img_path)

  # Save img to a tmp folder for training models.
  img_path = os.path.join(flags.save_img_path, str(uuid.uuid4()) + ".jpg")
  if not download_img(img_path):
    return

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
    resp = socket.recv()
    logging.info(resp)

    on_img_received(flags)
    input("wait for moving...")


if __name__ == "__main__":
  main()
