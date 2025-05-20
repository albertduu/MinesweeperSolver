import pytest
from utils.metrics import SolverMetrics, calculate_metrics

def test_metrics_calculation():
    metrics = SolverMetrics()
    metrics.wins = 4
    metrics.losses = 1
    metrics.decision_times = [0.1, 0.2, 0.15]
    metrics.mine_predictions = {
        (0,0): [0.2, 0.3],
        (1,1): [0.8, 0.9]
    }
    metrics.actual_mines = {(1,1)}
    
    # Add empty board history
    results = calculate_metrics(metrics, board_history=[])
    
    assert results['win_rate'] == 0.8
    assert results['mine_accuracy'] == 1.0
    assert pytest.approx(results['avg_decision_time']) == 0.15
    
