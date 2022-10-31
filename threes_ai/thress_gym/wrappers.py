from typing import Dict, List, NoReturn, Optional, Tuple, Union

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
        "card_type":
        spaces.MultiDiscrete(board_space),

        # Use a weight to indicate the relationship between numbers.
        "card_weight":
        spaces.Box(0., 1., shape=(1, BOARD_SIZE, BOARD_SIZE)),

        # Prior knowledge: a white card should merge with another card of same type.
        "is_white_card":
        spaces.MultiBinary((1, BOARD_SIZE, BOARD_SIZE)),

        # candidate card channels.
        "candidate_board":
        spaces.MultiDiscrete(board_space),
    })

  def observation(self, obs):
    board = torch.from_numpy(obs['board'])

    card_type = torch.zeros(self.board_shape, dtype=int)
    card_type[0, :, :] = board

    card_weight = torch.zeros(self.board_shape, dtype=torch.float32)
    card_weight[0, :, :] = board / TOTAL_STATE_NUM

    is_white_card = torch.zeros(self.board_shape, dtype=int)
    is_white_card[0, :, :] = (board >= 3).to(int)

    candidate_cards = obs['candidate_cards']
    candidate_board = torch.zeros(self.board_shape, dtype=int)
    for i, card in enumerate(candidate_cards):
      candidate_board[0, i, :] = card

    return {
        "card_type": card_type,
        "card_weight": card_weight,
        "is_white_card": is_white_card,
        "candidate_board": candidate_board,
    }


class VecEnv(gym.Env):

  def __init__(self, envs: List[gym.Env]):
    self.envs = envs
    self.last_outs = [() for _ in range(len(self.envs))]

  @staticmethod
  def _stack_dict(x: List[Union[Dict, np.ndarray]]) -> Union[Dict, np.ndarray]:
    if isinstance(x[0], dict):
      return {
          key: VecEnv._stack_dict([i[key] for i in x])
          for key in x[0].keys()
      }
    else:
      return np.stack([arr for arr in x], axis=0)

  @staticmethod
  def _vectorize_env_outs(env_outs: List[Tuple], reset: bool = False) -> Tuple:

    def _unzip_env_out():
      for env_out in env_outs:
        sz = len(env_out)
        if sz == 2:  # reset
          obs, info = env_out
          yield obs, 0, False, False, info
        else:
          # if env_out[2]:
          # print('[vec] terminated=', env_out[2], 'info=', env_out[-1])
          yield env_out

    obs_list, reward_list, done_list, _, info_list = zip(*_unzip_env_out())

    obs_stacked = VecEnv._stack_dict(obs_list)
    reward_stacked = np.array(reward_list)
    done_stacked = np.array(done_list)
    info_stacked = VecEnv._stack_dict(info_list)
    return obs_stacked, reward_stacked, done_stacked, info_stacked

  def reset(self, force: bool = False, **kwargs):
    if force:
      # noinspection PyArgumentList
      self.last_outs = [env.reset(**kwargs) for env in self.envs]
      return VecEnv._vectorize_env_outs(self.last_outs)

    for i, env in enumerate(self.envs):
      # Check if env finished
      if self.last_outs[i][2]:
        # noinspection PyArgumentList
        self.last_outs[i] = env.reset(**kwargs)
    return VecEnv._vectorize_env_outs(self.last_outs)

  def step(self, actions: torch.Tensor):
    assert len(actions) == len(
        self.envs), f"n_actions={len(actions)}, n_envs={len(self.envs)}"
    self.last_outs = [env.step(int(a)) for env, a in zip(self.envs, actions)]
    return VecEnv._vectorize_env_outs(self.last_outs)

  def render(self, idx: int, mode: str = None, **kwargs):
    # noinspection PyArgumentList
    return self.envs[idx].render(mode, **kwargs)

  def close(self):
    return [env.close() for env in self.envs]

  def seed(self, seed: Optional[int] = None) -> list:
    if seed is not None:
      return [env.seed(seed + i) for i, env in enumerate(self.envs)]
    else:
      return [env.seed(seed) for i, env in enumerate(self.envs)]

  @property
  def unwrapped(self) -> List[gym.Env]:
    return [env.unwrapped for env in self.envs]

  @property
  def action_space(self) -> List[gym.spaces.Dict]:
    return [env.action_space for env in self.envs]

  @property
  def observation_space(self) -> List[gym.spaces.Dict]:
    return [env.observation_space for env in self.envs]

  @property
  def metadata(self) -> List[Dict]:
    return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):

  def __init__(self,
               env: Union[gym.Env, VecEnv],
               device: torch.device = torch.device("cpu")):
    super(PytorchEnv, self).__init__(env)
    self.device = device

  def reset(self, **kwargs):
    return tuple([
        self._to_tensor(out, key_hint='reset')
        for out in super(PytorchEnv, self).reset(**kwargs)
    ])

  def step(self, action: torch.Tensor):
    assert len(action.shape) == 1
    return tuple([
        self._to_tensor(out, key_hint='step')
        for out in super(PytorchEnv, self).step(action)
    ])

  def _to_tensor(self,
                 x: Union[Dict, np.ndarray],
                 key_hint=None) -> Dict[str, Union[Dict, torch.Tensor]]:
    if isinstance(x, dict):
      return {
          key: self._to_tensor(val, key_hint=key)
          for key, val in x.items()
      }
    else:
      try:
        return torch.from_numpy(x).to(self.device, non_blocking=True)
      except Exception as e:
        print(key_hint)
        print(x)
        print(x.shape)
        raise e


class DictEnv(gym.Wrapper):

  @staticmethod
  def _dict_env_out(env_out: tuple) -> dict:
    obs, reward, done, info = env_out
    assert "obs" not in info.keys()
    assert "reward" not in info.keys()
    assert "done" not in info.keys()
    return dict(obs=obs, reward=reward, done=done, info=info)

  def reset(self, **kwargs):
    return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

  def step(self, action):
    v = DictEnv._dict_env_out(super(DictEnv, self).step(action))
    # print([(i, x) for i, x in enumerate(zip(v['obs'].items(), v['done']))])
    return v
