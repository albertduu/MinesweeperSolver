import numpy as np
from collections import defaultdict

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

def calculate_metrics(metrics, board_history):
    win_rate = metrics.wins / (metrics.wins + metrics.losses) if (metrics.wins + metrics.losses) > 0 else 0
    
    correct = 0
    total = 0
    for (x,y), preds in metrics.mine_predictions.items():
        is_mine = (y, x) in metrics.actual_mines
        avg_pred = np.mean(preds)
        if (avg_pred > 0.5 and is_mine) or (avg_pred <= 0.5 and not is_mine):
            correct += 1
        total += 1
    
    return {
        'win_rate': win_rate,
        'mine_accuracy': correct / total if total > 0 else 0,
        'avg_decision_time': np.mean(metrics.decision_times) if metrics.decision_times else 0,
        'moves_per_game': metrics.total_moves / (metrics.wins + metrics.losses) if (metrics.wins + metrics.losses) > 0 else 0,
        'total_games': metrics.wins + metrics.losses
    }

def track_mine_locations(board):
    return set(zip(*np.where(board.board == -1)))
