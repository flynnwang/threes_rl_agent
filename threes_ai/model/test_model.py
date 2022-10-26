
from threes_ai.thress_gym.env import ThreesEnv
from threes_ai.thress_gym.wrappers import ModelInputWrapper
from threes_ai.model import create_model

def test_model_init():
  wrapper = ModelInputWrapper(ThreesEnv())
  obs, _ = wrapper.reset()

  m = create_model(wrapper.observation_space, wrapper.action_space,
                   embedding_dim=16, hidden_dim=16, n_blocks=4)
  result = m(obs)
  assert 'policy_logits' in result
  assert 'baseline' in result

