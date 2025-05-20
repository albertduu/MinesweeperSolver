import numpy as np

class ProbabilisticSolver:
    def __init__(self, board):
        self.board = board
        self.probabilities = np.zeros((board.height, board.width))
        self.hidden = ~board.visible & ~board.flags
        self.update_probabilities()

    def update_probabilities(self):
        self.hidden = ~self.board.visible & ~self.board.flags
        mine_count = np.sum(self.board.flags)
        remaining_mines = self.board.num_mines - mine_count

        self.probabilities.fill(0.0)
        self.board.probabilities.fill(0.0)

        total_hidden = np.sum(self.hidden)
        if total_hidden == remaining_mines:
            self.probabilities[self.hidden] = 1.0
            self.board.probabilities[self.hidden] = 1.0
            return

        if remaining_mines == 0:
            self.probabilities[self.hidden] = 0.0
            self.board.probabilities[self.hidden] = 0.0
            return

        changed = True
        iteration_count = 0
        max_iterations = 100

        while changed and iteration_count < max_iterations:
            changed = False
            self.hidden = ~self.board.visible & ~self.board.flags
            iteration_count += 1

            for y in range(self.board.height):
                for x in range(self.board.width):
                    if not self.board.visible[y, x] or self.board.board[y, x] == 0:
                        continue

                    clue = self.board.board[y, x]
                    flagged = 0
                    hidden_adj = []

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.board.height and 0 <= nx < self.board.width:
                                if self.board.flags[ny, nx]:
                                    flagged += 1
                                elif self.hidden[ny, nx]:
                                    hidden_adj.append((ny, nx))

                    remaining = clue - flagged

                    if remaining == len(hidden_adj) and len(hidden_adj) > 0:
                        for ny, nx in hidden_adj:
                            if not self.board.flags[ny, nx]:
                                self.board.flags[ny, nx] = True
                                if self.probabilities[ny, nx] != 1.0:
                                    self.probabilities[ny, nx] = 1.0
                                    self.board.probabilities[ny, nx] = 1.0
                                    changed = True

                    elif remaining == 0 and len(hidden_adj) > 0:
                        for ny, nx in hidden_adj:
                            if self.probabilities[ny, nx] != 0.0:
                                self.probabilities[ny, nx] = 0.0
                                self.board.probabilities[ny, nx] = 0.0
                                changed = True

            self.hidden = ~self.board.visible & ~self.board.flags
            total_hidden = np.sum(self.hidden)
            remaining_mines = self.board.num_mines - np.sum(self.board.flags)

            if total_hidden == 0:
                return

            for y in range(self.board.height):
                for x in range(self.board.width):
                    if not self.hidden[y, x]:
                        continue

                    adjacent_clues = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < self.board.height and 0 <= nx < self.board.width and
                                self.board.visible[ny, nx] and self.board.board[ny, nx] > 0):
                                adjacent_clues.append((ny, nx))

                    if adjacent_clues:
                        prob_sum = 0
                        weight_sum = 0
                        for ny, nx in adjacent_clues:
                            clue = self.board.board[ny, nx]
                            hidden_around = 0
                            flagged_around = 0

                            for ddy in [-1, 0, 1]:
                                for ddx in [-1, 0, 1]:
                                    cy, cx = ny + ddy, nx + ddx
                                    if (0 <= cy < self.board.height and 0 <= cx < self.board.width):
                                        if self.board.flags[cy, cx]:
                                            flagged_around += 1
                                        elif self.hidden[cy, cx]:
                                            hidden_around += 1

                            remaining_clue = clue - flagged_around
                            if hidden_around > 0:
                                prob = remaining_clue / hidden_around
                                prob_sum += prob
                                weight_sum += 1

                        prob = min(prob_sum / weight_sum, 1.0) if weight_sum > 0 else remaining_mines / total_hidden
                    else:
                        prob = remaining_mines / total_hidden

                    self.probabilities[y, x] = prob
                    self.board.probabilities[y, x] = prob

    def next_move(self):
        self.update_probabilities()

        # print("Current board with probabilities:")
        # self.board.display(show_probabilities=True)

        safe_moves = np.argwhere((self.probabilities == 0.0) & self.hidden)
        for move in safe_moves:
            y, x = move
            if not self.board.visible[y, x]:
                # print(f"Choosing safe move at {(x, y)}")
                return (x, y)

        mine_moves = np.argwhere((self.probabilities == 1.0) & self.hidden)
        for move in mine_moves:
            y, x = move
            if not self.board.flags[y, x]:
                self.board.flags[y, x] = True
                # print(f"Flagging mine at {(x, y)}")
                return (x, y)

        revealed_tiles = np.argwhere(self.board.visible)
        adjacent_tiles = set()
        for tile in revealed_tiles:
            y, x = tile
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.board.height and 0 <= nx < self.board.width:
                        if self.hidden[ny, nx]:
                            adjacent_tiles.add((nx, ny))

        adjacent_tiles = list(adjacent_tiles)
        if adjacent_tiles:
            min_prob = min(self.probabilities[y, x] for x, y in adjacent_tiles)
            candidates = [(x, y) for x, y in adjacent_tiles if self.probabilities[y, x] == min_prob]
            if candidates:
                move = candidates[np.random.choice(len(candidates))]
                # print(f"Choosing lowest-risk move at {move} (probability: {min_prob:.0%})")
                return move

        hidden_cells = np.argwhere(self.hidden)
        if len(hidden_cells) > 0:
            for y, x in hidden_cells:
                if not self.board.visible[y, x]:
                    # print(f"Choosing fallback hidden cell at {(x, y)}")
                    return (x, y)

        return (0, 0)
