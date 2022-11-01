import torch

from threes_ai.thress_gym.env import ThreesEnv
from threes_ai.thress_gym.wrappers import ModelInputWrapper, VecEnv, PytorchEnv, DictEnv
from threes_ai.model import _create_model


def test_model_init():
  wrapper = ModelInputWrapper(ThreesEnv())

  m = _create_model(wrapper.observation_space,
                    wrapper.action_space,
                    embedding_dim=16,
                    hidden_dim=16,
                    n_blocks=4)

  env = VecEnv([wrapper])
  env = PytorchEnv(env, device=torch.device('cpu'))
  env = DictEnv(env)
  r = env.reset(force=True)
  result = m(r)
  # assert 'policy_logits' in result
  # assert 'baseline' in result
