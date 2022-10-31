import torch
from typing import Optional

from .env import ThreesEnv
from .wrappers import ModelInputWrapper, VecEnv, PytorchEnv, DictEnv


def create_game_env(seed=None):
  env = ThreesEnv(seed=seed)
  return ModelInputWrapper(env)


def create_env(flags,
               device: torch.device,
               seed: Optional[int] = None) -> DictEnv:
  if seed is None:
    seed = flags.seed
  envs = []
  for _ in range(flags.n_actor_envs):
    env = create_game_env()
    envs.append(env)
  env = VecEnv(envs)
  env = PytorchEnv(env, device)
  env = DictEnv(env)
  return env
