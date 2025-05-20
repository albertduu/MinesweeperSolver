import pytest
import numpy as np
from game.board import MinesweeperBoard

@pytest.fixture
def test_board():
    return MinesweeperBoard(width=5, height=5, num_mines=3)

def test_board_initialization(test_board):
    assert test_board.width == 5
    assert test_board.height == 5
    assert test_board.num_mines == 3
    assert test_board.game_active is True

def test_mine_generation(test_board):
    mine_count = np.sum(test_board.board == -1)
    assert mine_count == test_board.num_mines

def test_flood_fill_reveal(test_board):
    # Create valid board configuration with numbers
    test_board.board = np.array([
        [0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    test_board.reveal_cell(0, 0)
    assert test_board.visible[0, 0] == True
    assert test_board.visible[1, 1] == False  # Mine remains hidden

def test_win_condition(test_board):
    test_board.flags = test_board.board == -1
    test_board.visible = ~test_board.flags
    assert test_board.game_won() == True