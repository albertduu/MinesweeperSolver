import pytest
import numpy as np
from game.board import MinesweeperBoard
from solvers.probabilistic import ProbabilisticSolver

@pytest.fixture
def test_solver_board():
    board = MinesweeperBoard(width=5, height=5, num_mines=1)
    board.generate_board = lambda: None  # Disable auto-generation
    board.board = np.zeros((5,5), dtype=int)
    board.board[2,2] = -1
    # Set adjacent numbers
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            y = 2 + dy
            x = 2 + dx
            if 0 <= y < 5 and 0 <= x < 5:
                board.board[y][x] += 1
    board.visible = np.full((5,5), False)
    board.flags = np.full((5,5), False)
    return board

def test_probabilistic_solver_safe_moves(test_solver_board):
    # Set up a constrained scenario
    test_solver_board.board = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, -1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Reveal ALL cells except the mine
    for y in range(5):
        for x in range(5):
            if (y, x) != (2, 2):
                test_solver_board.reveal_cell(x, y)
    
    solver = ProbabilisticSolver(test_solver_board)
    solver.update_probabilities()
    
    # Should have 100% probability for the mine
    assert solver.probabilities[2, 2] == 1.0

def test_probabilistic_tie_breaking(test_solver_board, monkeypatch):
    solver = ProbabilisticSolver(test_solver_board)
    monkeypatch.setattr(solver, 'update_probabilities', lambda: None)
    
    # Setup hidden cells and probabilities
    solver.hidden = np.array([
        [True, True, False, False, False],
        [True, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
    ])
    solver.probabilities = np.array([
        [0.1, 0.1, 0.3, 0.3, 0.3],
        [0.1, 0.2, 0.3, 0.3, 0.3],
        [0.3, 0.3, 1.0, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3]
    ])
    
    move = solver.next_move()
    assert move in [(0,0), (0,1), (1,0)]