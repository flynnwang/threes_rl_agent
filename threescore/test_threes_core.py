
from threescore import (Board, BonusCards, Cell, Deck, MoveDirection,
                        ThreesGame, BOARD_SIZE)


def test_bonus_card_not_active():
  assert not BonusCards(24).is_active(), dir(BonusCards())
  assert not BonusCards(12).is_active()
  assert not BonusCards(6).is_active()
  assert not BonusCards(3).is_active()


def test_bonus_card_is_active():
  assert BonusCards(48).is_active()

  bc = BonusCards(384)
  assert bc.is_active()
  assert bc.get_active_bonus_cards() == [6, 12, 24, 48]


def test_deck():
  d = Deck()

  assert len(d.cards) == 12
  assert d.cards.count(1) == 4
  assert d.cards.count(2) == 4
  assert d.cards.count(3) == 4

  c = d.next()
  assert c in [1, 2, 3]
  assert d.cards.count(c) == 3

  for _ in range(11):
    d.next()

  assert d.empty(), d.cards


def test_deck_auto_refill():
  d = Deck()

  for _ in range(12):
    d.next()

  assert d.empty()

  card = d.next()
  assert card in [1, 2, 3]
  assert len(d) == 11
  assert d.cards.count(card) == 3


def make_board(cells):
  return [[Cell(cells[i][j])
           for j in range(BOARD_SIZE)]
          for i in range(BOARD_SIZE)]



def _check_move(b, dir, moved_cells, expected_idx):
  b_moved, dropin_positions = b.move(dir)
  b_expected = Board(make_board(moved_cells))
  print('moved=')
  print(b_moved)
  print('--')
  print('expected')
  print(b_expected)
  assert b_moved == b_expected
  assert dropin_positions == expected_idx


def test_board_move_case1():
  cells = [
    [2, 0, 0, 2],
    [0, 0, 3, 2],
    [1, 0, 1, 3],
    [0, 0, 1, 3],
  ]
  b = Board(make_board(cells))


  cells = [
    [2, 0, 3, 2],
    [1, 0, 1, 2],
    [0, 0, 1, 6],
    [0, 0, 0, 0],
  ]
  tail_idx = [(3, 0), (3, 1), (3, 2), (3, 3)]
  _check_move(b, MoveDirection.UP, cells, tail_idx)

  cells = [
    [0, 0, 0, 0],
    [2, 0, 3, 2],
    [0, 0, 1, 2],
    [1, 0, 1, 6],
  ]
  tail_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]
  _check_move(b, MoveDirection.DOWN, cells, tail_idx)


  cells = [
    [2, 0, 2, 0],
    [0, 3, 2, 0],
    [1, 1, 3, 0],
    [0, 1, 3, 0],
  ]
  tail_idx = [(0, 3), (1, 3), (2, 3), (3, 3)]
  _check_move(b, MoveDirection.LEFT, cells, tail_idx)

  cells = [
    [0, 2, 0, 2],
    [0, 0, 3, 2],
    [0, 1, 1, 3],
    [0, 0, 1, 3],
  ]
  tail_idx = [(0, 0), (1, 0), (2, 0), (3, 0)]
  _check_move(b, MoveDirection.RIGHT, cells, tail_idx)


def test_board_move_up():
  cells = [
    [0, 0, 1, 3],
    [0, 0, 0, 0],
    [3, 2, 0, 1],
    [1, 2, 1, 3],
  ]
  b = Board(make_board(cells))

  cells = [
    [0, 0, 1, 3],
    [3, 2, 0, 1],
    [1, 2, 1, 3],
    [0, 0, 0, 0],
  ]
  tail_idx = [(3, 0), (3, 1), (3, 2), (3, 3)]
  _check_move(b, MoveDirection.UP, cells, tail_idx)


def test_board_move_left():
  cells = [
    [0, 0, 1, 3],
    [3, 2, 0, 1],
    [1, 2, 1, 3],
    [0, 0, 0, 3],
  ]
  b = Board(make_board(cells))

  cells = [
    [0, 1, 3, 0],
    [3, 2, 1, 0],
    [3, 1, 3, 0],
    [0, 0, 3, 0],
  ]
  tail_idx = [(0, 3), (1, 3), (2, 3), (3, 3)]
  _check_move(b, MoveDirection.LEFT, cells, tail_idx)


def test_board_move_down():
  cells = [
    [0, 1, 3, 2],
    [3, 2, 1, 0],
    [3, 1, 3, 0],
    [0, 0, 3, 0],
  ]
  b = Board(make_board(cells))

  cells = [
    [0, 0, 0, 0],
    [0, 1, 3, 2],
    [3, 2, 1, 0],
    [3, 1, 6, 0],
  ]
  tail_idx = [(0, 0), (0, 1), (0, 2), (0, 3)]
  _check_move(b, MoveDirection.DOWN, cells, tail_idx)


def test_board_move_right():
  cells = [
    [0, 2, 0, 0],
    [0, 1, 3, 2],
    [3, 2, 1, 0],
    [3, 1, 6, 0],
  ]
  b = Board(make_board(cells))

  cells = [
    [0, 0, 2, 0],
    [0, 1, 3, 2],
    [0, 3, 2, 1],
    [0, 3, 1, 6],
  ]
  tail_idx = [(0, 0), (1, 0), (2, 0), (3, 0)]
  _check_move(b, MoveDirection.RIGHT, cells, tail_idx)


def test_game_start():
  game = ThreesGame()
  game.reset()

  assert game.board.count_card() == 9


def test_game_move_by_deck():
  class MockDeck:

    def next(self):
      return 6

  class MockBonusCard:

    def is_active(self):
      return False

  cells = [
    [0, 1, 3, 2],
    [3, 2, 1, 0],
    [3, 1, 3, 0],
    [0, 0, 3, 0],
  ]

  game = ThreesGame()
  game.board = Board(make_board(cells))
  game.deck = MockDeck()
  game.bonus_cards = MockBonusCard()

  game.move(MoveDirection.DOWN)

  cells = [
    [0, 0, 0, 0],
    [0, 1, 3, 2],
    [3, 2, 1, 0],
    [3, 1, 6, 0],
  ]
  expected_board = Board(make_board(cells))
  for i in range(1, BOARD_SIZE):
    for j in range(BOARD_SIZE):
      assert game.board.cells[i][j] == expected_board.cells[i][j]

  diff_count = 0
  dropin_card = -1
  for j in range(BOARD_SIZE):
    if game.board.cells[0][j] != expected_board.cells[0][j]:
      diff_count += 1
      dropin_card = game.board.cells[0][j].card
  assert diff_count == 1
  assert dropin_card == 6

