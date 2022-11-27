import argparse
import logging
import os
import time
import uuid
from typing import List, Optional
from pathlib import Path

import torch
import cv2
import numpy as np

from threes_ai.threes.core import ThreesGame, Board, Cell, NextCard, copy_board
from threes_ai.threes.consts import ACTION_TO_DIRECTION
from threes_ai.threes.img_utils import CardExtractor
from threes_ai.model.card import create_digit_model, predict_digit
from threes_ai.model import create_model
from threes_ai.thress_gym import create_test_env, create_game_env


def board_diff(observed_board, predicted_board, dropin_positions):
  for (r, c) in dropin_positions:
    ob = observed_board.get_card(r, c)
    pred = predicted_board.get_card(r, c)
    if ob != pred:
      return (r, c), True
  return None, False


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

    checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file,
                                  map_location=torch.device("cpu"))
    logging.info("Loading model parameters from checkpoint state...")
    self.actor_model.load_state_dict(checkpoint_state["model_state_dict"])
    self.actor_model.eval()
    logging.info("step handler ready!")

    self.game = None
    self.predictd_board = None
    self.dropin_positions = None

  def observe(self, img_path):
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

  def step(self, game):
    env = create_test_env(game, self.flags.actor_device)
    env_output = env.reset(force=True)
    agent_output = self.actor_model(env_output, sample=False)
    return agent_output["actions"][0]

  def wait_user_input(self, candi_cards, board):

    def numbers_to_line(nums):
      return ','.join([str(_) for _ in nums]) + '\n'

    with open(self.flags.manual_fix_path, 'w') as f:
      f.write(numbers_to_line(candi_cards))
      for i in range(4):
        numbers = [board.get_card(i, j) for j in range(4)]
        f.write(numbers_to_line(numbers))

    input('check on file: ' + self.flags.manual_fix_path + '\n')

    def line_to_numbers(line):
      return [int(x) for x in line.strip().split(',')]

    cells = copy_board(board.cells)
    with open(self.flags.manual_fix_path, 'r') as f:
      candi_cards = line_to_numbers(f.readline())

      for i in range(4):
        nums = line_to_numbers(f.readline())
        for j in range(4):
          cells[i][j].card = nums[j]
    return candi_cards, Board(cells)

  def execute(self, img_path, manual_fix=True):
    candi_cards, board = self.observe(img_path)

    pos = None
    game_board = board
    if self.game is not None:
      game_board = self.game.board

      # Fill in new cards if diff exists.
      pos, exists = board_diff(board, self.predictd_board, self.dropin_positions)
      if exists:
        x, y = pos
        self.game.board.cells[x][y].card = board.get_card(x, y)
        logging.info(f"Fill in new card at ({x}, {y}): {board.get_card(x, y)}")

    next_card = NextCard(game_board, candidate_cards=candi_cards)
    self.game = ThreesGame(game_board, next_card)
    self.game.display(start_pos=pos)

    if manual_fix:
      candi_cards, board = self.wait_user_input(candi_cards, game_board)
      next_card = NextCard(board, candidate_cards=candi_cards)
      self.game = ThreesGame(board, next_card)
      self.game.display(start_pos=pos)

    action = self.step(self.game)
    direction = ACTION_TO_DIRECTION[int(action)]
    logging.info(f"action: {int(action)}, move={direction}")

    # Attempt move and get dropin_positions.
    new_board, dropin_positions = self.game.board.move(direction)
    self.predictd_board = new_board
    self.dropin_positions = dropin_positions

    # Move board without dropin new cards.
    self.game.board = new_board
    return direction
