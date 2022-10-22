
import numpy as np

from .wrappers import ModelInputWrapper
from .env import ThreesEnv
from threes.consts import *

def test_model_input_wrapper():
  wrapper = ModelInputWrapper(None)

  board = np.array([[ 0,  7, 11,  9],
                    [12,  2,  3, 12],
                    [ 8, 10,  7, 11],
                    [11,  6, 13, 14]])
  candidate_cards = np.array([3, 7, 2])

  obs = wrapper.observation({'board': board, 'candidate_cards': candidate_cards})

  def weight(x):
    return (x + 1) / (TOTAL_STATE_NUM + 1)

  card_exists = obs['card_exists']
  card_weight = obs['card_weight']
  assert card_exists[0, 0, :, :].sum() == 1
  assert card_exists[0, 0, 0, 0] == 1
  assert np.isclose(card_weight[0, 0, 0, 0], weight(0))

  assert card_exists[0, 11, :, :].sum() == 3
  assert card_exists[0, 11, 0, 2] == 1
  assert card_exists[0, 11, 3, 0] == 1
  assert card_exists[0, 11, 2, 3] == 1
  assert np.isclose(card_weight[0, 11, 0, 2], weight(11))
  assert np.isclose(card_weight[0, 11, 3, 0], weight(11))
  assert np.isclose(card_weight[0, 11, 2, 3], weight(11))
  assert np.isclose(card_weight[0, 11, 0, 0], 0)

  assert card_exists[0, 12, :, :].sum() == 2
  assert card_exists[0, 12, 1, 0] == 1
  assert card_exists[0, 12, 1, 3] == 1
  assert np.isclose(card_weight[0, 12, 1, 0], weight(12))
  assert np.isclose(card_weight[0, 12, 1, 3], weight(12))

  is_white_card = obs['is_white_card']
  assert is_white_card.sum() == (16 - 2)
  assert is_white_card[0, 0, 0, 0] == 0
  assert is_white_card[0, 0, 1, 1] == 0
  assert is_white_card[0, 0, 3, 0] == 1
  assert is_white_card[0, 0, 2, 3] == 1

  is_candidate_cards = obs['is_candidate_cards']
  for c in candidate_cards:
    assert is_candidate_cards[0, c, :, :].sum() == 16
  assert is_candidate_cards[0, 0, :, :].sum() == 0
  assert is_candidate_cards[0, 8, :, :].sum() == 0
