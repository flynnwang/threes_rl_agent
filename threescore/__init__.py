import random
from enum import Enum, auto


BOARD_SIZE = 4


class Cell:

  def __init__(self, card=0):
    self.card = card

  def has_card(self):
    return self.card > 0

  def empty(self):
    return not self.has_card()

  def clear(self):
    self.card = 0

  def can_merge(self, c1):
    """Move from c1 => c2 (self)."""
    if c1.empty():
      return False
    if self.empty():
      return True
    s = c1.card + self.card
    return s == 3 or (s >= 6 and c1.card == self.card)

  def merge(self, c1):
    assert c1.has_card()
    self.card += c1.card

  def __repr__(self):
    if self.has_card():
      return f"<{self.card}>"
    return "<_>"

  def __eq__(self, other):
    if isinstance(other, Cell):
        return self.card == other.card
    return False


def make_board(size):
  return [[Cell() for i in range(size)]
          for _ in range(4)]

def copy_board(cells):
  return [[Cell(cells[i][j].card)
           for j in range(BOARD_SIZE)]
           for i in range(BOARD_SIZE)]


class MoveDirection(Enum):
  LEFT = auto()
  RIGHT = auto()
  UP = auto()
  DOWN = auto()


class Board:

  def __init__(self, cells=None):
    self.cells = cells
    if self.cells is None:
      self.cells = make_board(BOARD_SIZE)

  def can_move(self, direction: MoveDirection):
    new_board = make_board(BOARD_SIZE)
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        dx, dy = direction
        next_i = min(max(i + dx, 0), BOARD_SIZE-1)
        next_j = min(max(j + dy, 0), BOARD_SIZE-1)

  def get_units(self, direction: MoveDirection):
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        r, c = i, j
        if direction in (MoveDirection.RIGHT, MoveDirection.DOWN):
          c = BOARD_SIZE - c - 1
        if direction in (MoveDirection.UP, MoveDirection.DOWN):
          r, c = c, r
        yield r, c

  def move(self, direction: MoveDirection):
    """move rules:
    1. move if the next position is empty
    2. or two cards can merge"""

    cells = copy_board(self.cells)
    units = self.get_units(direction)
    tail_empty_cell_idx = []
    for _ in range(BOARD_SIZE):
      prev_cell = None
      prev_rc = None
      print('--')

      debug_cells = []
      for i in range(BOARD_SIZE):
        r, c = next(units)
        cur_cell = cells[r][c]
        print(r, c, cur_cell)

        if i != 0 and prev_cell.can_merge(cur_cell):
          # print(f'merge C<{r}, {c}>={cur_cell.card} into C<{prev_rc[0], prev_rc[1]}>={prev_cell.card}')
          prev_cell.merge(cur_cell)
          cur_cell.clear()

        if i == BOARD_SIZE - 1 and cur_cell.empty():
          tail_empty_cell_idx.append((r, c))

        prev_cell = cur_cell
        prev_rc = (r, c)
        debug_cells.append(cur_cell)

      print(debug_cells)
      print('--')

    return Board(cells), tail_empty_cell_idx

  def put(self, x, y, card: int):
    assert self.cells[x][y].empty()
    self.cells[x][y] = Cell(card)

  def __repr__(self):
    rows = []
    for i in range(BOARD_SIZE):
      rows.append(str(self.cells[i]))
    return '\n'.join(rows)



class Deck:

  def __init__(self):
    self.cards = [1] * 4 + [2] * 4 + [3] * 4

  def __repr__(self):
    return str(self.cards)

  def empty(self):
    return len(self.cards) == 0

  def next(self):
    card = random.choice(self.cards)
    self.cards.remove(card)
    return card

class BonusCards:

  def __init__(self, max_card_val):
    self.max_card_val = max_card_val

  def is_active(self):
    return self.max_card_val >= 48

  def get_active_bonus_cards(self):
    max_bonus_card = self.max_card_val // 8

    def _gen():
      c = 6
      while c <= max_bonus_card:
        yield c
        c *= 2
    return list(_gen())


class ThreesGame:

  def __init__(self):
    self.board = Board()
    self.deck = Deck()

  def done(self):
    pass

  def move_up():
    pass
