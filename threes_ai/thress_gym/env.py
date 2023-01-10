import random
from typing import Dict, List, NoReturn, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces

from threes_ai.threes.consts import *
from threes_ai.threes import ThreesGame

# DIRECTION_TO_ACTION = {v: k for k, v in ACTION_TO_DIRECTION.items()}

CARD_TO_STATE = dict((c, i) for i, c in enumerate(CARDS))


class ThreesEnv(gym.Env):
  metadata = {"render_modes": ["console"], "render_fps": 4}

  def __init__(self, render_mode=None, seed=None):
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    self._seed = seed
    self.game = None

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
    self._actions_taken_mask = np.zeros(NUM_ACTIONS, dtype=np.int32)

  def _get_obs(self):
    board_obs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        card = self.game.board.get_card(i, j)
        assert card in CARDS, f"what? {card}"
        board_obs[i][j] = CARD_TO_STATE[card]

    candidate_cards_obs = np.zeros(MAX_CANDIDATE_CARDS_NUM, dtype=int)
    candidate_cards = self.game.next_card.peek()
    for i, card in enumerate(candidate_cards):
      candidate_cards_obs[i] = CARD_TO_STATE[card]

    return {"board": board_obs, "candidate_cards": candidate_cards_obs}

  def _get_action_mask(self):
    moves = self.game.get_available_moves()
    actions_mask = np.zeros(NUM_ACTIONS, dtype=np.int32)
    for a, m in ACTION_TO_DIRECTION.items():
      if m in moves:
        actions_mask[a] = 1
    return actions_mask

  def _get_info(self):
    available_action_mask = self._get_action_mask()
    info = {
        # action mask after taking current action, and available for current board state.
        'available_action_mask': available_action_mask,

        # action mask before taking/selecting current action
        'actions_taken_mask': self._actions_taken_mask,
        'game_step_count': self.game.num_step,
    }

    board_info = self.game.board.get_info()
    info.update(board_info)
    return info

  def seed(self, seed: Optional[int] = None) -> NoReturn:
    self._seed = seed

  def reset(self, seed=None):
    seed = seed or self._seed
    random.seed(seed)
    super().reset(seed=seed)

    self.game = ThreesGame()
    self.game.reset()

    obs = self._get_obs()
    info = self._get_info()
    return obs, info

  def step(self, action: int):
    # print('before info', self._get_info())
    # self.game.display()

    # compute info first to get accurate actions_taken_mask.
    self._actions_taken_mask = self._get_action_mask()
    num_card_before_moving = self.game.board.count_card()

    direction = ACTION_TO_DIRECTION[action]
    moved = self.game.move(direction)

    terminated = self.game.done()

    obs = self._get_obs()
    info = self._get_info()

    # num_merged_card = 0
    # num_card_after_moving = info['num_card']

    # if moved:
    # # for a successful move, one card will be added to the board.
    # num_card_after_moving -= 1
    # reward = num_merged_card
    # r1 = num_merged_card / 50.0

    # reward /= 500.0  # Given that we're targeting this max score

    # TARGET_CARD = 1536
    TARGET_CARD = 6144
    def reward_score(r):
      # if r < TARGET_CARD:
        # return 0
      return (r / TARGET_CARD)

    merge_sum = self.game.board.merge_sum
    reward = reward_score(merge_sum) if merge_sum > 0 else 0

    # max_card = self.game.board.max_card()
    # if terminated and max_card < THE_MAX_CARD:
      # reward -= reward_score(self.game.board.max_card()) / 2
      # if max_card < TARGET_CARD:
        # reward -= 1
      # else:
        # reward -= reward_score(self.game.board.max_card()) / 2

    # empty_before = (16 - num_card_before_moving)
    # empty_after = (16 - num_card_after_moving)
    # num_lost_cells = (empty_before - empty_after)

    # penality = 0
    # if num_lost_cells > 0:
    # penality = -reward_score(num_lost_cells * 3)
    # reward += penality
    # print(
    # f"lost_cells={num_lost_cells}, penality={penality}, card_before={num_card_before_moving}, card_after={num_card_after_moving}"
    # )

    # print("num_merged=%s, r1=%s, r2=%s, merge_sum=%s" %
    # (num_merged_card, r1, reward, merge_sum))
    # debug
    # if num_merged_card == 0 and merge_sum > 0:
    # self.game.display()
    # print('after info: ', info)
    # print('action: ', action, ACTION_TO_DIRECTION[action])
    # __import__('ipdb').set_trace()

    # actions_taken_mask = info['actions_taken_mask']
    # if actions_taken_mask[action] == 0:
    # __import__('ipdb').set_trace()
    # print()

    # if terminated:
    # print('terminated=', terminated, 'info=', info)
    # observation, reward, terminated, truncated, info
    return obs, reward, terminated, False, info

  def render(self):
    if self.render_mode == "console":
      self.game.display()


class ThreesObservedEnv(ThreesEnv):

  def step(self, action: int):
    raise NotImplementedError

  def reset(self):
    assert self.game is not None, "game object should not be None."

    obs = self._get_obs()
    info = self._get_info()
    return obs, info
