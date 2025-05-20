import tensorflow as tf
import numpy as np
from game.board import MinesweeperBoard

class CNNSolver:
    def __init__(self, difficulty='beginner'):
        model_path = self._get_model_path(difficulty)
        self.model = tf.keras.models.load_model(model_path)

    def _get_model_path(self, difficulty):
        difficulty = difficulty.lower()
        if difficulty == 'beginner':
            return 'models/cnn_beginner.keras'
        elif difficulty == 'intermediate':
            return 'models/cnn_intermediate.keras'
        elif difficulty == 'expert':
            return 'models/cnn_expert.keras'
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")

    def get_move(self, board: MinesweeperBoard):
        if board.first_move:
            # Make a safe first move
            return board.width // 2, board.height // 2, 0.0

        input_tensor = self._create_input_tensor(board)
        prediction = self.model.predict(tf.expand_dims(input_tensor, axis=0), verbose=0)[0, :, :, 0]

        # Mask already visible or flagged cells
        masked_prediction = np.copy(prediction)
        masked_prediction[board.visible] = np.inf
        masked_prediction[board.flags] = np.inf
        # print("Raw predictions:", prediction.min(), prediction.max(), prediction.mean())
        # Choose the cell with the lowest probability of being a mine
        y, x = np.unravel_index(np.argmin(masked_prediction), masked_prediction.shape)
        confidence = np.clip(1 + float(masked_prediction[y, x]), 0.001, 0.999)
        return x, y, confidence

    def _create_input_tensor(self, board: MinesweeperBoard) -> tf.Tensor:
        channels = np.zeros((board.height, board.width, 11), dtype=np.float32)

        # Channel 0: Visible
        channels[:, :, 0] = board.visible.astype(float)

        # Channel 1: Flags
        channels[:, :, 1] = board.flags.astype(float)

        # Channels 2-10: One-hot numbers
        for i in range(9):
            channels[:, :, 2+i] = (board.board == i).astype(float)

        return tf.convert_to_tensor(channels, dtype=tf.float32)
