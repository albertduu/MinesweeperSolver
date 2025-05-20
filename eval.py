import time
import argparse
import numpy as np
from collections import defaultdict
from solvers.cnn import CNNSolver
from solvers.probabilistic import ProbabilisticSolver
from game.board import MinesweeperBoard

class SolverMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.wins = 0
        self.losses = 0
        self.total_moves = 0
        self.decision_times = []
        self.mine_predictions = defaultdict(list)
        self.actual_mines = set()

def track_mine_locations(board):
    return set(zip(*np.where(board.board == -1)))

def run_headless_simulation(solver_type, width, height, num_mines, iterations=1, difficulty="beginner"):
    all_metrics = []
    for _ in range(iterations):
        board = MinesweeperBoard(width, height, num_mines)
        metrics = SolverMetrics()
        metrics.reset()
        metrics.actual_mines = track_mine_locations(board)
        
        if solver_type == "cnn":
            solver = CNNSolver(difficulty=difficulty)
        else:
            solver = ProbabilisticSolver(board)
        
        while board.game_active and not board.game_won():
            start_time = time.time()
            if solver_type == "cnn":
                x, y, confidence = solver.get_move(board)
            else:
                x, y = solver.next_move()
                confidence = 1.0 - board.probabilities[y][x]
            decision_time = time.time() - start_time
            
            mine_prediction = 1 - confidence
            metrics.mine_predictions[(x, y)].append(mine_prediction)
            metrics.decision_times.append(decision_time)
            metrics.total_moves += 1
            
            mine_hit = board.reveal_cell(x, y)
            if mine_hit:
                metrics.losses += 1
                break
        
        if board.game_won():
            metrics.wins += 1
        else:
            metrics.losses += 1
        
        all_metrics.append(metrics)
    
    return all_metrics

def aggregate_metrics(all_metrics):
    total_wins = sum(m.wins for m in all_metrics)
    total_games = len(all_metrics)
    win_rate = total_wins / total_games if total_games > 0 else 0
    
    total_correct = 0
    total_preds = 0
    for metrics in all_metrics:
        for (x, y), preds in metrics.mine_predictions.items():
            is_mine = (y, x) in metrics.actual_mines
            avg_pred = np.mean(preds) if preds else 0
            if (avg_pred > 0.5 and is_mine) or (avg_pred <= 0.5 and not is_mine):
                total_correct += 1
            total_preds += 1
    mine_accuracy = total_correct / total_preds if total_preds > 0 else 0
    
    decision_times = []
    for m in all_metrics:
        decision_times.extend(m.decision_times)
    avg_decision_time = np.mean(decision_times) if decision_times else 0
    
    total_moves = sum(m.total_moves for m in all_metrics)
    moves_per_game = total_moves / total_games if total_games > 0 else 0
    
    return {
        'win_rate': win_rate,
        'mine_accuracy': mine_accuracy,
        'avg_decision_time': avg_decision_time,
        'moves_per_game': moves_per_game,
        'total_games': total_games
    }

def compare_solvers(iterations=10):
    standard_configs = [
        {'name': 'beginner', 'width': 9, 'height': 9, 'mines': 10},
        {'name': 'intermediate', 'width': 16, 'height': 16, 'mines': 40},
        {'name': 'expert', 'width': 30, 'height': 16, 'mines': 99}
    ]
    
    for config in standard_configs:
        print(f"\n=== Testing {config['name']} ({config['width']}x{config['height']}, {config['mines']} mines) ===")
        
        print("Running CNN solver...")
        cnn_metrics = aggregate_metrics(
            run_headless_simulation("cnn", config['width'], config['height'], config['mines'], iterations, config['name'])
        )
        
        print("Running Probabilistic solver...")
        prob_metrics = aggregate_metrics(
            run_headless_simulation("probabilistic", config['width'], config['height'], config['mines'], iterations, config['name'])
        )
        
        print(f"\n{config['name']} Results:")
        print("CNN Solver:")
        print(f"  Win Rate: {cnn_metrics['win_rate']:.1%}")
        print(f"  Mine Accuracy: {cnn_metrics['mine_accuracy']:.1%}")
        print(f"  Avg Decision Time: {cnn_metrics['avg_decision_time']:.4f}s")
        print(f"  Moves/Game: {cnn_metrics['moves_per_game']:.1f}")
        
        print("\nProbabilistic Solver:")
        print(f"  Win Rate: {prob_metrics['win_rate']:.1%}")
        print(f"  Mine Accuracy: {prob_metrics['mine_accuracy']:.1%}")
        print(f"  Avg Decision Time: {prob_metrics['avg_decision_time']:.4f}s")
        print(f"  Moves/Game: {prob_metrics['moves_per_game']:.1f}")
        print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch compare Minesweeper solvers across standard configurations')
    parser.add_argument("--iterations", type=int, default=9,
                       help="Number of games to run per configuration")
    args = parser.parse_args()
    
    compare_solvers(args.iterations)