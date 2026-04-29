"""
othello_game.py
Othello Game Engine — Team Othello
ICSI435/535 Artificial Intelligence — University at Albany

Handles board state, legal move generation, disc flipping,
forced pass detection, and game-over logic.
"""

import numpy as np

# Board values
EMPTY = 0
BLACK = 1
WHITE = -1

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
CORNERS = [(0,0),(0,7),(7,0),(7,7)]


class OthelloGame:
    def __init__(self):
        self.board = self.create_board()
        self.current_player = BLACK
        self.move_count = 0
        self.game_over = False

    def create_board(self):
        board = np.zeros((8, 8), dtype=int)
        board[3][3] = WHITE
        board[3][4] = BLACK
        board[4][3] = BLACK
        board[4][4] = WHITE
        return board

    def get_legal_moves(self, board, player):
        opponent = -player
        legal = []
        for r in range(8):
            for c in range(8):
                if board[r][c] != EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                        nr += dr
                        nc += dc
                        found_opponent = True
                    if found_opponent and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                        legal.append((r, c))
                        break
        return legal

    def make_move(self, board, row, col, player):
        board = board.copy()
        board[row][col] = player
        opponent = -player
        for dr, dc in DIRECTIONS:
            to_flip = []
            nr, nc = row + dr, col + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                to_flip.append((nr, nc))
                nr += dr
                nc += dc
            if to_flip and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                for (fr, fc) in to_flip:
                    board[fr][fc] = player
        return board

    def get_score(self, board):
        black = int(np.sum(board == BLACK))
        white = int(np.sum(board == WHITE))
        return black, white

    def get_winner(self, board):
        black, white = self.get_score(board)
        if black > white:
            return BLACK
        elif white > black:
            return WHITE
        return 0

    def is_game_over(self, board):
        return (not self.get_legal_moves(board, BLACK) and
                not self.get_legal_moves(board, WHITE))

    def step(self, row, col):
        legal = self.get_legal_moves(self.board, self.current_player)
        if (row, col) not in legal:
            return False
        self.board = self.make_move(self.board, row, col, self.current_player)
        self.move_count += 1
        self.current_player = -self.current_player
        if not self.get_legal_moves(self.board, self.current_player):
            self.current_player = -self.current_player
        if self.is_game_over(self.board):
            self.game_over = True
        return True

    def reset(self):
        self.__init__()
