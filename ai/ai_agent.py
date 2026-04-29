"""
ai_agent.py
-----------
Minimax AI with Alpha-Beta Pruning and move ordering.
Includes node counter and timer for time complexity measurement.

Team Othello - ICSI435/535 Artificial Intelligence
University at Albany
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.heuristics import evaluate, get_phase, get_legal_moves_for, PHASE_WEIGHTS

BLACK =  1
WHITE = -1
DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def make_move_on_board(board, row, col, player):
    """Execute move and return new board (does not modify original)."""
    import numpy as np
    opponent = -player
    new_board = board.copy()
    new_board[row][col] = player
    for dr, dc in DIRECTIONS:
        to_flip = []
        nr, nc = row + dr, col + dc
        while 0 <= nr < 8 and 0 <= nc < 8 and new_board[nr][nc] == opponent:
            to_flip.append((nr, nc))
            nr += dr; nc += dc
        if to_flip and 0 <= nr < 8 and 0 <= nc < 8 and new_board[nr][nc] == player:
            for fr, fc in to_flip:
                new_board[fr][fc] = player
    return new_board


class OthelloAI:
    """
    Othello AI using Minimax with Alpha-Beta Pruning.

    Features:
    - Configurable search depth
    - Move ordering for pruning efficiency
    - Node counter for complexity measurement
    - Timer for performance measurement
    - Pluggable heuristic (multi-factor or single-factor)
    """

    def __init__(self, player, depth=5, heuristic="multi", custom_weights=None):
        """
        Args:
            player:         BLACK (1) or WHITE (-1)
            depth:          Search depth (3-7 recommended)
            heuristic:      'multi' | 'coin_parity' | 'mobility' | 'corners' |
                            'positional' | 'stability'
            custom_weights: Optional dict to override phase weights
        """
        self.player         = player
        self.depth          = depth
        self.heuristic      = heuristic
        self.custom_weights = custom_weights

        # ── Measurement counters ──────────────────────────────
        self.nodes_explored  = 0
        self.last_move_time  = 0.0
        self.nodes_per_depth = {}  # depth → node count

    def reset_counters(self):
        self.nodes_explored = 0
        self.last_move_time = 0.0

    def _evaluate(self, board, player):
        """Route to correct evaluation function based on heuristic setting."""
        from ai.heuristics import evaluate_single_factor
        if self.heuristic == "multi":
            return evaluate(board, player, custom_weights=self.custom_weights)
        else:
            return evaluate_single_factor(board, player, self.heuristic)

    def _order_moves(self, moves, board, player):
        """
        Sort moves best-first using a shallow evaluation.
        Critical for maximising Alpha-Beta pruning efficiency.
        Without ordering: ~30% reduction.
        With ordering:    ~50% reduction.
        """
        scored = []
        for r, c in moves:
            new_board = make_move_on_board(board, r, c, player)
            score = self._evaluate(new_board, player)
            scored.append((score, (r, c)))
        scored.sort(reverse=True)
        return [move for _, move in scored]

    # ── Minimax with Alpha-Beta Pruning ───────────────────────
    def _alphabeta(self, board, depth, alpha, beta, is_maximizing, player):
        """
        Minimax with Alpha-Beta Pruning.

        Time complexity: O(b^d) worst case → O(b^(d/2)) with good move ordering
        where b = branching factor (~10-20 in Othello), d = search depth

        Args:
            board:          Current board state
            depth:          Remaining depth to search
            alpha:          Best score maximizer can guarantee
            beta:           Best score minimizer can guarantee
            is_maximizing:  True if current node is MAX (AI's turn)
            player:         The AI's player value (1 or -1)

        Returns:
            float: Best evaluation score for this subtree
        """
        self.nodes_explored += 1  # ← Node counter for complexity measurement

        # ── Terminal conditions ───────────────────────────────
        if depth == 0:
            return self._evaluate(board, player)

        current_player = player if is_maximizing else -player
        moves = get_legal_moves_for(board, current_player)

        if not moves:
            # No legal moves — forced pass
            opponent_moves = get_legal_moves_for(board, -current_player)
            if not opponent_moves:
                # Game over — evaluate final position
                return self._evaluate(board, player)
            # Pass turn
            return self._alphabeta(board, depth - 1, alpha, beta, not is_maximizing, player)

        # Order moves for better pruning
        moves = self._order_moves(moves, board, current_player)

        if is_maximizing:
            best = float('-inf')
            for r, c in moves:
                new_board = make_move_on_board(board, r, c, current_player)
                score = self._alphabeta(new_board, depth - 1, alpha, beta, False, player)
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break  # ← Prune — beta cut-off
            return best
        else:
            best = float('inf')
            for r, c in moves:
                new_board = make_move_on_board(board, r, c, current_player)
                score = self._alphabeta(new_board, depth - 1, alpha, beta, True, player)
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break  # ← Prune — alpha cut-off
            return best

    def get_best_move(self, board):
        """
        Find the best move for self.player from current board.

        Returns:
            tuple: (row, col) of best move, or None if no moves available
        """
        self.reset_counters()
        start_time = time.time()

        moves = get_legal_moves_for(board, self.player)
        if not moves:
            return None

        moves = self._order_moves(moves, board, self.player)

        best_score = float('-inf')
        best_move  = moves[0]

        for r, c in moves:
            new_board = make_move_on_board(board, r, c, self.player)
            score = self._alphabeta(
                new_board, self.depth - 1,
                float('-inf'), float('inf'),
                False, self.player
            )
            if score > best_score:
                best_score = score
                best_move  = (r, c)

        self.last_move_time = time.time() - start_time
        return best_move

    def get_stats(self):
        """Return measurement stats after get_best_move() call."""
        return {
            "nodes_explored": self.nodes_explored,
            "move_time_sec":  round(self.last_move_time, 4),
            "depth":          self.depth,
            "heuristic":      self.heuristic,
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"[AI Stats] Depth: {s['depth']} | "
              f"Nodes: {s['nodes_explored']:,} | "
              f"Time: {s['move_time_sec']:.4f}s | "
              f"Heuristic: {s['heuristic']}")
