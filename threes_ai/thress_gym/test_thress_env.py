from .env import ThreesEnv


def test_three_env():
  env = ThreesEnv()
  obs, info = env.reset(seed=42)
  for _ in range(10):
    obs, reward, done, _, info = env.step(env.action_space.sample())
