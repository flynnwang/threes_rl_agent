import random

from .consts import *

INITIAL_CARD_NUM = 9

BONUS_CARD_CHANCE = 1. / 21.


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
  return [[Cell() for i in range(size)] for _ in range(4)]


def copy_board(cells):
  return [[Cell(cells[i][j].card) for j in range(BOARD_SIZE)]
          for i in range(BOARD_SIZE)]


class Board:

  def __init__(self, cells=None):
    self.cells = cells
    if self.cells is None:
      self.cells = make_board(BOARD_SIZE)

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
    dropin_positions = []
    for _ in range(BOARD_SIZE):
      prev_cell = None
      prev_rc = None
      # print('--')

      debug_cells = []
      for i in range(BOARD_SIZE):
        r, c = next(units)
        cur_cell = cells[r][c]
        # print(r, c, cur_cell)

        if i != 0 and prev_cell.can_merge(cur_cell):
          # print(f'merge C<{r}, {c}>={cur_cell.card} into C<{prev_rc[0], prev_rc[1]}>={prev_cell.card}')
          prev_cell.merge(cur_cell)
          cur_cell.clear()

        if i == BOARD_SIZE - 1 and cur_cell.empty():
          dropin_positions.append((r, c))

        prev_cell = cur_cell
        prev_rc = (r, c)
        debug_cells.append(cur_cell)

      # print(debug_cells)
      # print('--')

    return Board(cells), dropin_positions

  def put(self, x, y, card: int):
    assert self.cells[x][y].empty()
    self.cells[x][y] = Cell(card)

  def count_card(self):
    c = 0
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        if self.cells[i][j].has_card():
          c += 1
    return c

  def max_card(self):
    card = -1
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        if self.cells[i][j].has_card():
          card = max(card, self.cells[i][j].card)
    return card

  def get_card(self, x, y):
    return self.cells[x][y].card

  def __eq__(self, other):
    if isinstance(other, Board):
      return self.cells == other.cells
    return False

  def __repr__(self):
    rows = []
    for i in range(BOARD_SIZE):
      rows.append(str(self.cells[i]))
    return '\n'.join(rows)


class Deck:

  def __init__(self):
    self.cards = self._refill()
    self.next_idx = None

  def _refill(self):
    return [1] * 4 + [2] * 4 + [3] * 4

  def __repr__(self):
    return str(self.cards)

  def empty(self):
    return len(self.cards) == 0

  def peek(self):
    if self.empty():
      self.cards = self._refill()

    if self.next_idx is None:
      self.next_idx = random.randint(0, len(self.cards) - 1)

    return self.cards[self.next_idx]

  def next(self, peek=False):
    card = self.peek()
    self.cards.pop(self.next_idx)
    self.next_idx = None
    return card

  def __len__(self):
    return len(self.cards)


class BonusCards:

  def __init__(self, max_card):
    self.max_card = max_card

  def is_active(self):
    return self.max_card >= 48

  def use_bonus_card(self):
    return random.random() < BONUS_CARD_CHANCE

  def get_active_bonus_cards(self):
    max_bonus_card = self.max_card // 8

    def _gen():
      c = 6
      while c <= max_bonus_card:
        yield c
        c *= 2

    return list(_gen())

  def gen_candidate_cards(self):
    assert self.is_active()

    cards = self.get_active_bonus_cards()
    n = min(len(cards), MAX_CANDIDATE_CARDS_NUM)

    candidate_cards = random.sample(cards, n)
    candidate_cards.sort()
    return candidate_cards

  def update(self, max_card):
    self.max_card = max_card


class NextCard:

  def __init__(self, board):
    self.board = board
    self.deck = Deck()
    self.bonus_cards = BonusCards(board.max_card())
    self.candidate_cards = None

  def peek(self):
    if self.candidate_cards is not None:
      return self.candidate_cards

    max_card = self.board.max_card()
    self.bonus_cards.update(max_card)

    if self.bonus_cards.is_active() and self.bonus_cards.use_bonus_card():
      self.candidate_cards = self.bonus_cards.gen_candidate_cards()
    else:
      self.candidate_cards = [self.deck.next()]

    return self.candidate_cards

  def next(self):
    candidate_cards = self.peek()
    self.candidate_cards = None
    return random.choice(candidate_cards)


class ThreesGame:

  def __init__(self, board=None):
    self.board = board
    self.next_card = None

  def reset(self):
    self.board = Board()
    self.next_card = NextCard(self.board)
    self._fill_initial_board()

  def done(self):
    """Game is over if all four move directions are dead."""
    for dir in list(MoveDirection):
      _, dropin_positions = self.board.move(dir)
      if dropin_positions:
        return False
    return True

  def peek(self):
    return self.board, self.next_card.peek()

  def move(self, direction: MoveDirection):
    new_board, dropin_positions = self.board.move(direction)
    if not dropin_positions:
      return False

    new_card = self.next_card.next()
    pos = random.choice(dropin_positions)
    new_board.put(*pos, new_card)

    self.board = new_board
    return True

  def _fill_initial_board(self):
    positions = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
    positions = random.sample(positions, INITIAL_CARD_NUM)

    for i, j in positions:
      card = self.next_card.next()
      self.board.put(i, j, card)

  def display(self):
    max_card = self.board.max_card()
    width = len(str(abs(max_card)))
    fmt = "{0: <%s}" % width

    if self.done():
      print("DONE :P")
    print(f"Next card(s): {self.next_card.peek()}")
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        c = self.board.cells[i][j]
        if c.has_card():
          print(fmt.format(c.card), end='')
        else:
          print('_' * width, end='')
        print('  ', end='')
      print()