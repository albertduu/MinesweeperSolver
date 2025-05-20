import pygame
import time
import sys
import argparse
from solvers.cnn import CNNSolver
from solvers.probabilistic import ProbabilisticSolver
from game.board import MinesweeperBoard
from configs import WIDTH, HEIGHT, NUM_MINES
from game.gui import MinesweeperGUI
from utils.metrics import SolverMetrics, calculate_metrics, track_mine_locations

def run_gui_game(solver_type):
    pygame.init()
    gui = MinesweeperGUI(WIDTH, HEIGHT, cell_size=40)

    board = MinesweeperBoard(WIDTH, HEIGHT, NUM_MINES)

    if solver_type == "cnn":
        solver = CNNSolver()
    elif solver_type == "probabilistic":
        solver = ProbabilisticSolver(board)
    else:
        print(f"Unknown solver type: {solver_type}")
        return

    gui.update_board(board)
    gui.update_display()

    steps = 0
    running = True
    while running and board.game_active and not board.game_won():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if solver_type == "cnn":
            x, y, confidence = solver.get_move(board)
            print(f"Step {steps}: Trying ({x}, {y}) with confidence {confidence:.2f}")
        else:
            x, y = solver.next_move()
            confidence = 1.0 - board.probabilities[y, x]
            print(f"Step {steps}: Trying ({x}, {y}) with confidence {confidence:.2f}")

        gui.highlight_move(x, y, confidence)
        gui.update_display()
        time.sleep(0.8)

        mine_hit = board.reveal_cell(x, y)
        gui.update_board(board)
        gui.update_display()
        steps += 1

        if mine_hit:
            print("Mine hit! Game over.")
            break
        if board.game_won():
            print("Game won!")

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["cnn", "probabilistic"], default="cnn", help="Choose which solver to use")
    args = parser.parse_args()

    run_gui_game(args.solver)
