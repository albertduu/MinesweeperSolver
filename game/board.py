import numpy as np
import random
from itertools import product

class MinesweeperBoard:
    def __init__(self, width=9, height=9, num_mines=10):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = np.zeros((height, width), dtype=int)
        self.visible = np.full((height, width), False)
        self.flags = np.full((height, width), False)
        self.game_active = True
        self.first_move = True
        self.probabilities = np.zeros((self.height, self.width))


    def place_mines(self, safe_x, safe_y):
        positions = [(x, y) for x, y in product(range(self.width), range(self.height)) 
                    if not (x == safe_x and y == safe_y)]
        
        mine_positions = random.sample(positions, self.num_mines)
        for x, y in mine_positions:
            self.board[y, x] = -1

        for y in range(self.height):
            for x in range(self.width):
                if self.board[y, x] == -1:
                    continue
                self.board[y, x] = self._count_adjacent_mines(x, y)

    def _count_adjacent_mines(self, x, y):
        count = 0
        for dx, dy in product([-1, 0, 1], repeat=2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                count += self.board[ny, nx] == -1
        return count

    def reveal_cell(self, x, y):
        if not self.game_active or self.flags[y, x]:
            return False

        if self.first_move:
            self.place_mines(x, y)
            self.first_move = False

        if self.board[y, x] == -1:
            self.game_active = False
            self.visible[y, x] = True
            return True

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if not self.visible[y, x]:
                self.visible[y, x] = True
                if self.board[y, x] == 0:
                    for dx, dy in product([-1, 0, 1], repeat=2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if not (dx == 0 and dy == 0) and self.board[ny, nx] != -1:
                                stack.append((nx, ny))
        return False

    def toggle_flag(self, x, y):
        if self.game_active and not self.visible[y, x]:
            self.flags[y, x] = not self.flags[y, x]

    def get_random_safe_cell(self):
        safe_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.visible[y, x] and not self.flags[y, x] and self.board[y, x] != -1:
                    safe_cells.append((x, y))
        if not safe_cells:
            raise ValueError("No safe cells remaining")
        return random.choice(safe_cells)

    def game_won(self):
        return np.sum(self.visible) == (self.width * self.height - self.num_mines)

    def display(self, show_probabilities=False):
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.flags[y, x]:
                    row.append('F')
                elif not self.visible[y, x]:
                    row.append('.')
                else:
                    row.append(str(self.board[y, x]) if self.board[y, x] != -1 else '*')
            print(' '.join(row))
        print()
