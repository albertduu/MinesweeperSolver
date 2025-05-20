import pygame
import numpy as np
from pygame.locals import *

COLORS = {
    0: (200, 200, 200),   # Empty
    1: (0, 0, 255),       # 1
    2: (0, 128, 0),       # 2
    3: (255, 0, 0),       # 3
    4: (0, 0, 128),       # 4
    -1: (0, 0, 0),        # Mine
    'hidden': (150, 150, 150),
    'flag': (255, 0, 0),
    'highlight': (255, 255, 0),
    'grid': (100, 100, 100),
    'text': (0, 0, 0),
    'background': (255, 255, 255)
}

class MinesweeperGUI:
    def __init__(self, width, height, cell_size=40):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Initialize display first
        self.screen = pygame.display.set_mode(
            (width * cell_size, height * cell_size))
        pygame.display.set_caption("Minesweeper Solver")
        
        # Initialize font system
        pygame.font.init()
        
        # Now create fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.message_font = pygame.font.Font(None, 32)
        
        # Rest of initialization...
        self.current_move = None
        self.current_confidence = 0.0
        self.message = ""
        self.board = None

    def update_board(self, board):
        self.board = board
        self.draw_board()

    def draw_board(self):
        self.screen.fill(COLORS['background'])
        if not self.board:
            return
            
        # Draw grid lines
        for x in range(self.board.width + 1):
            pygame.draw.line(self.screen, COLORS['grid'], 
                            (x * self.cell_size, 0), 
                            (x * self.cell_size, self.board.height * self.cell_size), 1)
        for y in range(self.board.height + 1):
            pygame.draw.line(self.screen, COLORS['grid'], 
                            (0, y * self.cell_size), 
                            (self.board.width * self.cell_size, y * self.cell_size), 1)

        # Draw cells
        for y in range(self.board.height):
            for x in range(self.board.width):
                rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                                  self.cell_size-1, self.cell_size-1)
                
                if self.board.visible[y, x]:
                    if self.board.board[y, x] == -1:
                        pygame.draw.rect(self.screen, COLORS[-1], rect)
                    else:
                        pygame.draw.rect(self.screen, COLORS[0], rect)
                        if self.board.board[y, x] > 0:
                            text = self.font.render(str(self.board.board[y, x]), True, 
                                                  COLORS[self.board.board[y, x]])
                            self.screen.blit(text, (x*self.cell_size+10, y*self.cell_size+10))
                else:
                    pygame.draw.rect(self.screen, COLORS['hidden'], rect)
                    if self.board.flags[y, x]:
                        pygame.draw.circle(self.screen, COLORS['flag'], 
                                         (x*self.cell_size+20, y*self.cell_size+20), 8)

        # Draw current move highlight and confidence
        if self.current_move:
            x, y = self.current_move
            rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                              self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, COLORS['highlight'], rect, 3)
            
            # Confidence text
            r = max(0, min(255, int(255 * (1 - self.current_confidence))))
            g = max(0, min(255, int(255 * self.current_confidence)))
            prob_color = (r, g, 0)
            prob_text = f"{self.current_confidence*100:.1f}%"
            text = self.small_font.render(prob_text, True, prob_color)
            text_rect = text.get_rect(center=(x*self.cell_size+self.cell_size//2, 
                                         y*self.cell_size+self.cell_size//2))
            # self.screen.blit(text, text_rect)

        # Status message
        if self.message:
            text = self.message_font.render(self.message, True, COLORS['text'])
            text_rect = text.get_rect(center=(self.width*self.cell_size//2, 
                                           self.height*self.cell_size//2))
            self.screen.blit(text, text_rect)

    def highlight_move(self, x, y, confidence):
        self.current_move = (x, y)
        self.current_confidence = confidence
        self.draw_board()

    def show_explosion(self, x, y):
        rect = pygame.Rect(x*self.cell_size, y*self.cell_size,
                          self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), rect)
        self.update_display()

    def show_message(self, text, persist=False):
        self.message = text
        self.draw_board()
        if not persist:
            self.message = ""

    def update_display(self):
        pygame.display.flip()

    def reset(self):
        self.current_move = None
        self.current_confidence = 0.0
        self.message = ""
        self.board = None
        self.draw_board()
        self.update_display()