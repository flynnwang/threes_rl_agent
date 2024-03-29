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
    return s == 3 or (c1.card == self.card and 6 <= s <= THE_MAX_CARD)

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

  def __init__(self, cells=None, merge_sum=0):
    self.cells = cells
    if self.cells is None:
      self.cells = make_board(BOARD_SIZE)
    self.merge_sum = merge_sum

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
    merge_sum = 0
    for _ in range(BOARD_SIZE):
      prev_cell = None
      moved = False
      # pr, pc = -1, -1
      for i in range(BOARD_SIZE):
        r, c = next(units)
        cur_cell = cells[r][c]
        # print(r, c, cur_cell)

        if i != 0 and prev_cell.can_merge(cur_cell):
          if prev_cell.has_card() and cur_cell.has_card():
            merge_sum += prev_cell.card + cur_cell.card
            # print('action=%s, merge (%s, %s) %s => (%s, %s) %s' %
            # (direction, r, c, cur_cell, pr, pc, prev_cell))

          prev_cell.merge(cur_cell)
          cur_cell.clear()
          moved = True

        if i == BOARD_SIZE - 1 and cur_cell.empty() and moved:
          dropin_positions.append((r, c))

        prev_cell = cur_cell
        # pr = r
        # pc = c

    return Board(cells, merge_sum), dropin_positions

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

  def has_nearby_6144s(self):
    for i in range(BOARD_SIZE - 1):
      for j in range(BOARD_SIZE - 1):
        c = self.cells[i][j]
        if c.card != THE_MAX_CARD:
          continue

        c1 = self.cells[i][j + 1]
        if c1.card == THE_MAX_CARD:
          return True

        c2 = self.cells[i + 1][j]
        if c2.card == THE_MAX_CARD:
          return True
    return False

  def get_info(self):
    mx_card = -1
    num_card = 0
    card_sum = 0
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        if self.cells[i][j].has_card():
          card = self.cells[i][j].card
          mx_card = max(mx_card, card)
          num_card += 1
          card_sum += card
    return {
        'max_card': mx_card,
        'num_card': num_card,
        'card_sum': card_sum,
    }

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

  def __init__(self, game):
    self.game = game

  @property
  def max_card(self):
    return self.game.board.max_card()

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
    if len(cards) < MAX_CANDIDATE_CARDS_NUM:
      return cards

    # drop first and last card then sample 1 card.
    cards = cards[1:-1]
    center_card = random.choice(cards)
    return [center_card // 2, center_card, center_card * 2]


class NextCard:

  def __init__(self, game, candidate_cards=None):
    self.deck = Deck()
    self.bonus_cards = BonusCards(game)
    self.candidate_cards = candidate_cards

  def peek(self):
    if self.candidate_cards is not None:
      return self.candidate_cards

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

  def __init__(self, board=None, next_card=None):
    self.board = board
    self.next_card = next_card
    self.num_step = 0

  def reset(self):
    self.board = Board()
    self.next_card = NextCard(self)
    self._fill_initial_board()

  def get_available_moves(self):
    moves = set()
    for dir in list(MoveDirection):
      _, dropin_positions = self.board.move(dir)
      if dropin_positions:
        moves.add(dir)
    return moves

  def done(self):
    """Game is over if all four move directions are dead."""
    return (len(self.get_available_moves()) == 0
            or self.board.max_card() == THE_MAX_CARD)

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
    self.num_step += 1
    return True

  def _fill_initial_board(self):
    positions = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
    positions = random.sample(positions, INITIAL_CARD_NUM)

    add_6144 = True
    add_3072 = True
    for idx, (i, j) in enumerate(positions):
      card = self.next_card.next()

      # adding a 6144 to make a harder game.
      if idx == 0 and add_6144:
        card = 6144
      if idx == 1 and add_3072:
        card = 3072

      self.board.put(i, j, card)

  def display(self, start_pos=None):
    max_card = self.board.max_card()
    width = len(str(abs(max_card)))
    fmt = "{0: <%s}" % width

    if self.done():
      print("DONE :P")
    print(f"Next card(s): {self.next_card.peek()}")
    for i in range(BOARD_SIZE):
      for j in range(BOARD_SIZE):
        c = self.board.cells[i][j]
        is_new_card = start_pos and (i, j) == start_pos

        if is_new_card:
          print('[', end='')

        # print card
        card_str = ""
        if c.has_card():
          card_str = fmt.format(c.card)
        else:
          card_str = '_' * width
        print(card_str, end='')

        if is_new_card:
          print(']', end='')
        else:
          print('  ', end='')
      print()
