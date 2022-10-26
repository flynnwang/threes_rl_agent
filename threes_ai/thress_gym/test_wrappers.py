
import torch
import numpy as np

from threes_ai.thress_gym.wrappers import ModelInputWrapper
from threes_ai.threes.consts import *
from .env import ThreesEnv

def test_model_input_wrapper():
  wrapper = ModelInputWrapper(None)

  board = np.array([[ 0,  7, 11,  9],
                    [12,  2,  3, 12],
                    [ 8, 10,  7, 11],
                    [11,  6, 13, 14]])
  candidate_cards = np.array([3, 7, 2])

  obs = wrapper.observation({'board': board, 'candidate_cards': candidate_cards})

  def weight(x):
    return (x) / (TOTAL_STATE_NUM)

  card_type = obs['card_type']
  card_weight = obs['card_weight']

  for i in range(BOARD_SIZE):
    for j in range(BOARD_SIZE):
      assert card_type[0, i, j] == board[i, j]
      assert np.isclose(card_weight[0, i, j], weight(board[i, j]))

  is_white_card = obs['is_white_card']
  assert is_white_card.sum() == (16 - 2)
  assert is_white_card[0, 0, 0] == 0
  assert is_white_card[0, 1, 1] == 0
  assert is_white_card[0, 3, 0] == 1
  assert is_white_card[0, 2, 3] == 1

  cb = obs['candidate_board']
  assert cb.shape == (1, 4, 4)
  assert torch.allclose(cb[0, 0, :],  torch.tensor([3]))
  assert torch.allclose(cb[0, 1, :],  torch.tensor([7]))
  assert torch.allclose(cb[0, 2, :],  torch.tensor([2]))
