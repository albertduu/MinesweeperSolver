import numpy as np
import random
import tensorflow as tf
from game.board import MinesweeperBoard

def create_random_field(width: int, height: int, num_mines: int) -> MinesweeperBoard:
    board = MinesweeperBoard(width, height, num_mines)
    
    num_sweeps = random.randrange(5, 25)
    for _ in range(num_sweeps):
        if not board.game_active:
            break
        try:
            x, y = board.get_random_safe_cell()
            mine_hit = board.reveal_cell(x, y)
            if mine_hit:
                board.game_active = False
        except ValueError:
            break
    return board

def create_probability_tensor(board: MinesweeperBoard) -> tf.Tensor:
    tensor = np.zeros((board.height, board.width), np.float32)
    mine_locations = (board.board == -1)
    tensor[mine_locations] = 1.0
    tensor = tensor.reshape(board.height, board.width, 1)
    return tf.convert_to_tensor(tensor, dtype=tf.float32)

def create_input_tensor(board: MinesweeperBoard) -> tf.Tensor:
    channels = np.zeros((board.height, board.width, 11), dtype=np.float32)

    channels[:, :, 0] = board.visible.astype(float)
    channels[:, :, 1] = board.flags.astype(float)

    for i in range(9):
        channels[:, :, 2 + i] = (board.board == i).astype(float)

    return tf.convert_to_tensor(channels, dtype=tf.float32)

def generate_training_data(width: int, height: int, num_mines: int, num_examples: int):
    input_tensors = []
    output_tensors = []
    for _ in range(num_examples):
        board = create_random_field(width, height, num_mines)
        input_tensors.append(create_input_tensor(board))
        output_tensors.append(create_probability_tensor(board))
    return tf.stack(input_tensors, axis=0), tf.stack(output_tensors, axis=0)
