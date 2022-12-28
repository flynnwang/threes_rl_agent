from enum import Enum, auto

BOARD_SIZE = 4

MAX_CANDIDATE_CARDS_NUM = 3

N_WHITE_CARD = 13

# Total number of cards + an empty card
CARDS = [0, 1, 2] + [3 * (2**i) for i in range(N_WHITE_CARD)]

THE_MAX_CARD = max(CARDS)

TOTAL_STATE_NUM = len(CARDS)  # 15

NUM_ACTIONS = 4


class MoveDirection(Enum):
  LEFT = auto()
  RIGHT = auto()
  UP = auto()
  DOWN = auto()


ACTION_TO_DIRECTION = dict(enumerate(list(MoveDirection)))
