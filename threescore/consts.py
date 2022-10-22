from enum import Enum, auto

BOARD_SIZE = 4

MAX_CANDIDATE_CARDS_NUM = 3

N_WHITE_CARD = 12

# Total number of cards + an empty card
CARDS = [0, 1, 2] + [3 * (2**i) for i in range(N_WHITE_CARD)]

MAX_CARD = max(CARDS)

TOTAL_STATE_NUM = len(CARDS) # 15

NUM_ACTIONS = 4
class MoveDirection(Enum):
  LEFT = auto()
  RIGHT = auto()
  UP = auto()
  DOWN = auto()
