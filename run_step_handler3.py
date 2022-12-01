import logging
import argparse
import time
import uuid
import os
import random

import requests
import hydra
from omegaconf import OmegaConf, DictConfig
from playsound import playsound

from threes_ai.threes.consts import MoveDirection
from threes_ai.threes.hanlder import StepHandler

handler = None

IMAGE_URL = "http://192.168.31.207:5000/board.jpg"
# IMAGE_URL = "http://127.0.0.1:5000/board.jpg"

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
    logging.info("Image downloaded: %s", img_path)
    return True
  logging.error("Failed to download image file!")
  return False


def request_game_state(flags):
  # Save img to a tmp folder for training models.
  img_path = os.path.join(flags.save_img_path, str(uuid.uuid4()) + ".jpg")
  if not download_img(img_path):
    return

  global handler
  if handler is None:
    handler = StepHandler(flags)

  direction = handler.execute(img_path)
  random_sound(direction)


@hydra.main(config_path="conf", config_name="step_handler_config")
def main(flags: DictConfig):
  if not os.path.exists(flags.save_img_path):
    os.makedirs(flags.save_img_path)

  while True:
    request_game_state(flags)
    input("wait for moving...")


if __name__ == "__main__":
  main()
