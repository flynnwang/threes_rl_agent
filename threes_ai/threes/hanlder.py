import argparse
import logging
import os
import time
import uuid
from typing import List, Optional

import cv2
import numpy as np

from threes_ai.threes.core import ThreesGame, Board, Cell, NextCard
from threes_ai.threes.consts import ACTION_TO_DIRECTION
from threes_ai.threes.img_utils import CardExtractor
from threes_ai.model.card import create_digit_model, predict_digit
from threes_ai.model import create_model
from threes_ai.thress_gym import create_test_env, create_game_env

# @dataclass
# class StepState:

# board_cards: List[List[int]]
# candi_cards: List[int]


class StepHandler:

  def __init__(self, flags):
    self.flags = flags

    logging.info("digit model loading...")
    self.digit_model = create_digit_model(flags.digit_model_ck_path,
                                          device=flags.actor_device)
    self.digit_model.eval()

    logging.info("actor model loading...")
    game_env = create_game_env()
    self.actor_model = create_model(flags, game_env, flags.actor_device)
    self.actor_model.eval()
    logging.info("step handler ready!")

  def _observe(self, img_path):
    ce = CardExtractor(img_path)
    candi_imgs, board_imgs = ce.extract()

    candi_nums = []
    for _, img in enumerate(candi_imgs):
      cat, _ = predict_digit(self.digit_model, img)
      candi_nums.append(int(cat))

    cells = []
    for i in range(4):
      row = []
      for j in range(4):
        img = board_imgs[i][j]
        cat, _ = predict_digit(self.digit_model, img)
        row.append(Cell(int(cat)))
      cells.append(row)
    return candi_nums, Board(cells)

  def _step(self, game):
    env = create_test_env(game, self.flags.actor_device)
    env_output = env.reset(force=True)
    agent_output = self.actor_model(env_output, sample=False)
    return agent_output["actions"][0]

  def execute(self, img_path):
    candi_cards, board = self._observe(img_path)
    next_card = NextCard(board, candidate_cards=candi_cards)
    game = ThreesGame(board, next_card)

    action = self._step(game)
    move = ACTION_TO_DIRECTION[int(action)]

    game.display()
    logging.info(f"action: {int(action)}, move={move}")
