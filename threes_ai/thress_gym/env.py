import random
from typing import Dict, List, NoReturn, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces

from threes_ai.threes.consts import *
from threes_ai.threes import ThreesGame

ACTION_TO_DIRECTION = dict(enumerate(list(MoveDirection)))

CARD_TO_STATE = dict((c, i) for i, c in enumerate(CARDS))


class ThreesEnv(gym.Env):
  metadata = {"render_modes": ["console"], "render_fps": 4}

  def __init__(self, render_mode=None, seed=None):
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    self._seed = seed

    # For each board position, the state could be one of the card or empty.
    # Same as the candidate card positions.
    board_space = np.zeros((BOARD_SIZE, BOARD_SIZE)) + TOTAL_STATE_NUM
    candidate_cards_space = np.zeros(MAX_CANDIDATE_CARDS_NUM) + TOTAL_STATE_NUM
    self.observation_space = spaces.Dict({
        "board":
        spaces.MultiDiscrete(board_space),
        "candidate_cards":
        spaces.MultiDiscrete(candidate_cards_space)
    })

    # LEFT/RIGHT/UP/DOWN
    self.action_space = spaces.Discrete(NUM_ACTIONS)

  def _get_obs(self):
    board_obs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        card = self.game.board.get_card(i, j)
        assert card in CARDS
        board_obs[i][j] = CARD_TO_STATE[card]

    candidate_cards_obs = np.zeros(MAX_CANDIDATE_CARDS_NUM, dtype=int)
    candidate_cards = self.game.next_card.peek()
    for i, card in enumerate(candidate_cards):
      candidate_cards_obs[i] = card

    return {"board": board_obs, "candidate_cards": candidate_cards_obs}

  def _get_info(self):
    return {'max_card': self.game.board.max_card()}

  def seed(self, seed: Optional[int] = None) -> NoReturn:
    self._seed = seed

  def reset(self, seed=None):
    seed = seed or self._seed
    super().reset(seed=seed)

    self.game = ThreesGame()
    self.game.reset()

    obs = self._get_obs()
    info = self._get_info()
    return obs, info

  def step(self, action: int):
    direction = ACTION_TO_DIRECTION[action]
    moved = self.game.move(direction)

    terminated = self.game.done()
    reward = 1 if moved else -1  # move as long as possible
    obs = self._get_obs()
    info = self._get_info()

    # if terminated:
    # print('terminated=', terminated, 'info=', info)
    # observation, reward, terminated, truncated, info
    return obs, reward, terminated, False, info

  def render(self):
    if self.render_mode == "console":
      self.game.display()
