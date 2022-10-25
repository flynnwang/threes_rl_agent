

import gym
import numpy as np
import torch
from gym import spaces

from threes_ai.threes.consts import *


class ModelInputWrapper(gym.ObservationWrapper):
  """Input Layers:

  board states:
  1. for each card (including 0), 1 = exist, 0 = not exists
  2. for each card a number between [0, 1], weight = n / max_card
  3. is_white_card: to indicate whether this card should merge with another same card

  candidate cards states:
  1. one layer for each card to indate it's a candidate card.
    (should exclude 0, and the largest 3 cards)
  """

  def __init__(self, env):
    super().__init__(env)

    self.board_shape = (1, BOARD_SIZE, BOARD_SIZE)
    board_space = torch.zeros(self.board_shape) + TOTAL_STATE_NUM
    self.observation_space = spaces.Dict({
      "card_type": spaces.MultiDiscrete(board_space),

      # Use a weight to indicate the relationship between numbers.
      "card_weight": spaces.Box(0., 1., shape=(1, BOARD_SIZE, BOARD_SIZE)),

      # Prior knowledge: a white card should merge with another card of same type.
      "is_white_card": spaces.MultiBinary((1, BOARD_SIZE, BOARD_SIZE)),

      # candidate card channels.
      "candidate_card_1": spaces.MultiDiscrete(board_space),
      "candidate_card_2": spaces.MultiDiscrete(board_space),
      "candidate_card_3": spaces.MultiDiscrete(board_space),
    })

  def observation(self, obs):
    board = torch.from_numpy(obs['board'])

    card_type = torch.zeros(self.board_shape, dtype=int)
    card_type[0, :, :] = board

    card_weight = torch.zeros(self.board_shape, dtype=torch.float32)
    card_weight[0, :, :] = board / TOTAL_STATE_NUM

    is_white_card = torch.zeros(self.board_shape, dtype=int)
    is_white_card[0, :, :] = (board >= 3).to(int)

    candidate_cards = torch.from_numpy(obs['candidate_cards'])
    def make_candidate_card_at(i):
      c = torch.zeros(self.board_shape, dtype=int)
      c[0, :, :] = candidate_cards[i]
      return c

    return {
      "card_type": card_type,
      "card_weight": card_weight,
      "is_white_card": is_white_card,

      "candidate_card_1": make_candidate_card_at(0),
      "candidate_card_2": make_candidate_card_at(1),
      "candidate_card_3": make_candidate_card_at(2),
    }
