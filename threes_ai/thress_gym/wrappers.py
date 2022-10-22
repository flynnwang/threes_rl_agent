

import gym
import numpy as np
from gym import spaces

from threes.consts import *


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

    self.observation_space = spaces.Dict({
      # For each type of card, check its existence on the board.
      "card_exists": spaces.MultiBinary((1, TOTAL_STATE_NUM, BOARD_SIZE, BOARD_SIZE)),

      # Use a weight to indicate the relationship between numbers.
      "card_weight": spaces.Box(0., 1., shape=(1, TOTAL_STATE_NUM, BOARD_SIZE, BOARD_SIZE)),

      # Prior knowledge: a white card should merge with another card of same type.
      "is_white_card": spaces.MultiBinary((1, 1, BOARD_SIZE, BOARD_SIZE)),

      # candidate cards channel.
      "is_candidate_cards": spaces.MultiBinary((1, TOTAL_STATE_NUM, BOARD_SIZE, BOARD_SIZE)),
    })

  def observation(self, obs):
    board = obs['board']
    candidate_cards = obs['candidate_cards']

    card_exists = np.zeros((1, TOTAL_STATE_NUM, BOARD_SIZE, BOARD_SIZE),
                           dtype=int)
    card_weight = np.zeros((1, TOTAL_STATE_NUM, BOARD_SIZE, BOARD_SIZE),
                           dtype=float)
    is_white_card = np.zeros((1, 1, BOARD_SIZE, BOARD_SIZE), dtype=int)
    for card_idx in range(TOTAL_STATE_NUM):
      exists = (board == card_idx).astype(int)
      card_exists[0, card_idx, :, :] = exists

      weight = exists * ((card_idx + 1) / (TOTAL_STATE_NUM + 1))
      card_weight[0, card_idx, :, :] = weight
    is_white_card[0, 0, :, :] = (board >= 3).astype(int)

    is_candidate_cards = np.zeros((1, TOTAL_STATE_NUM-1, BOARD_SIZE, BOARD_SIZE), dtype=int)
    for card_idx in candidate_cards:
      is_candidate_cards[0, card_idx, :, :] = 1

    return {"card_exists": card_exists,
            "card_weight": card_weight,
            "is_white_card": is_white_card,
            "is_candidate_cards": is_candidate_cards}
