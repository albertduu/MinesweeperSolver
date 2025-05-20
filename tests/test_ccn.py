import pytest
import numpy as np
from solvers.cnn import CNNSolver
from game.board import MinesweeperBoard

def test_cnn_preprocessing():
    board = MinesweeperBoard(width=3, height=3, num_mines=1)
    board.board = np.array([
        [-1, 1, 0],
        [ 1, 1, 0],
        [ 0, 0, 0]
    ])
    board.visible[0,1] = True  # Reveal a cell
    board.flags[0,0] = True    # Flag a mine
    
    solver = CNNSolver("dummy_path")
    processed = solver.preprocess_board(board)
    
    # Test visible channel
    assert processed[0,:,:,0].tolist() == [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    
    # Test flags channel
    assert processed[0,:,:,1].tolist() == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    
    # Test numbers channel
    assert processed[0,0,1,2] == 1.0/8  # Normalized number