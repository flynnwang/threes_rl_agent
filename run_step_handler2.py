import logging
import argparse
import time
import uuid
import os
import shutil
import random

import requests
import cv2
import numpy as np
import zmq
import hydra
from omegaconf import OmegaConf, DictConfig
from playsound import playsound

from threes_ai.threes.consts import MoveDirection
from threes_ai.threes.hanlder import StepHandler

handler = None

# IMAGE_URL = "http://192.168.31.207:8000/board.jpg"
IMAGE_URL = "http://169.254.112.56:8000/board.jpg"

SOUND_DIR = "/Users/flynn.wang/repo/flynn/thress_imgs/move_sound"


def random_sound(dir: MoveDirection):
  dir = str(dir).split('.')[-1]
  sound_dir = os.path.join(SOUND_DIR, dir)

  sound_files = os.listdir(sound_dir)
  sound_files = [
      os.path.join(SOUND_DIR, dir, x) for x in sound_files if x.endswith("m4a")
  ]
  sound_file = random.choice(sound_files)
  playsound(sound_file)


def download_img(img_path):
  r = requests.get(IMAGE_URL)
  if r.status_code == 200:
    img_data = r.content
    with open(img_path, 'wb') as f:
      f.write(img_data)
      # shutil.copyfileobj(img_data, f)
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

  manual_fix = False
  while True:
    direction = handler.execute(img_path, manual_fix=manual_fix)
    random_sound(direction)
    user_cmd = input("wait for user input (or leave it empty to move):")
    if user_cmd.strip() == "":
      handler.move(direction)
      break

    handler.game = None
    manual_fix = True


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


if __name__ == "__main__":
  main()
