"""
othello_game.py
---------------
Core Othello game engine.
Board values: 1 = Black, -1 = White, 0 = Empty
Winner values: 1 = Black wins, -1 = White wins, 0 = Draw

Team Othello - ICSI435/535 Artificial Intelligence
University at Albany
"""

import numpy as np
import copy

# ── Constants ─────────────────────────────────────────────────
BLACK   =  1
WHITE   = -1
EMPTY   =  0
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


class OthelloGame:
    """Full Othello game state and rule enforcement."""

    def __init__(self):
        self.board = self._init_board()
        self.current_player = BLACK   # Black always moves first
        self.move_count = 0

    # ── Board setup ───────────────────────────────────────────
    def _init_board(self):
        board = np.zeros((8, 8), dtype=int)
        board[3][3] = WHITE
        board[3][4] = BLACK
        board[4][3] = BLACK
        board[4][4] = WHITE
        return board

    # ── Legal move generation ─────────────────────────────────
    def get_legal_moves(self, player=None):
        """Return list of (row, col) tuples that are legal for player."""
        if player is None:
            player = self.current_player
        opponent = -player
        legal = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] != EMPTY:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opponent:
                        nr += dr
                        nc += dc
                        found_opponent = True
                    if found_opponent and 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == player:
                        legal.append((r, c))
                        break
        return legal

    # ── Move execution ────────────────────────────────────────
    def make_move(self, row, col, player=None):
        """Execute a move. Returns new board state (does not modify in place)."""
        if player is None:
            player = self.current_player
        opponent = -player
        new_board = self.board.copy()
        new_board[row][col] = player
        for dr, dc in DIRECTIONS:
            to_flip = []
            nr, nc = row + dr, col + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and new_board[nr][nc] == opponent:
                to_flip.append((nr, nc))
                nr += dr
                nc += dc
            if to_flip and 0 <= nr < 8 and 0 <= nc < 8 and new_board[nr][nc] == player:
                for fr, fc in to_flip:
                    new_board[fr][fc] = player
        return new_board

    def apply_move(self, row, col):
        """Apply move to self.board and advance the turn."""
        self.board = self.make_move(row, col, self.current_player)
        self.move_count += 1
        self._next_turn()

    def _next_turn(self):
        """Switch player, handle forced pass, detect game over."""
        self.current_player = -self.current_player
        if not self.get_legal_moves():
            # Opponent has no moves — switch back
            self.current_player = -self.current_player
            if not self.get_legal_moves():
                # Neither player can move — game over
                self.current_player = None

    # ── Game state queries ────────────────────────────────────
    def is_game_over(self):
        return self.current_player is None

    def get_score(self):
        black = int(np.sum(self.board == BLACK))
        white = int(np.sum(self.board == WHITE))
        return black, white

    def get_winner(self):
        """Returns 1 (Black wins), -1 (White wins), 0 (Draw)."""
        black, white = self.get_score()
        if black > white:
            return BLACK
        elif white > black:
            return WHITE
        return 0

    def get_filled_count(self):
        return int(np.count_nonzero(self.board))

    def get_phase(self):
        """Return game phase string based on filled squares."""
        filled = self.get_filled_count()
        if filled < 20:
            return "early"
        elif filled < 44:
            return "mid"
        else:
            return "end"

    def clone(self):
        """Return a deep copy of the game state."""
        new_game = OthelloGame.__new__(OthelloGame)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def __repr__(self):
        symbols = {BLACK: "B", WHITE: "W", EMPTY: "."}
        rows = []
        for r in range(8):
            rows.append(" ".join(symbols[self.board[r][c]] for c in range(8)))
        black, white = self.get_score()
        return "\n".join(rows) + f"\nBlack: {black}  White: {white}"
