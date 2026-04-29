"""
ai_agent.py
AI Agent — Minimax with Alpha-Beta Pruning — Team Othello
ICSI435/535 Artificial Intelligence — University at Albany

Implements:
    - Minimax search (baseline)
    - Alpha-Beta Pruning with move ordering
    - Node counter and timer for benchmarking
    - Configurable depth and heuristic mode
"""

import time
from othello_game import BLACK, WHITE, EMPTY
from heuristics import evaluate, evaluate_single_factor


class AIAgent:
    def __init__(self, player, depth=5, use_alpha_beta=True,
                 heuristic_mode='combined', single_factor=None):
        """
        Args:
            player:          BLACK (1) or WHITE (-1)
            depth:           search depth (3–7 recommended)
            use_alpha_beta:  True = Alpha-Beta, False = pure Minimax
            heuristic_mode:  'combined' | 'single' | 'coin_only'
            single_factor:   if heuristic_mode='single', which factor to use
        """
        self.player = player
        self.depth = depth
        self.use_alpha_beta = use_alpha_beta
        self.heuristic_mode = heuristic_mode
        self.single_factor = single_factor

        # Benchmarking counters
        self.nodes_explored = 0
        self.last_move_time = 0.0

    def reset_counters(self):
        self.nodes_explored = 0
        self.last_move_time = 0.0

    def get_best_move(self, board, game):
        """
        Entry point: returns the best (row, col) move for the current position.
        Also records nodes_explored and last_move_time.
        """
        self.reset_counters()
        legal_moves = game.get_legal_moves(board, self.player)
        if not legal_moves:
            return None

        start = time.time()
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            new_board = game.make_move(board, move[0], move[1], self.player)
            if self.use_alpha_beta:
                score = self._alphabeta(
                    new_board, self.depth - 1,
                    float('-inf'), float('inf'),
                    False, game
                )
            else:
                score = self._minimax(new_board, self.depth - 1, False, game)

            if score > best_score:
                best_score = score
                best_move = move

        self.last_move_time = time.time() - start
        return best_move

    def _eval(self, board, game):
        """Call the appropriate evaluation function."""
        self.nodes_explored += 1
        if self.heuristic_mode == 'coin_only':
            from heuristics import coin_parity
            return coin_parity(board, self.player)
        elif self.heuristic_mode == 'single' and self.single_factor:
            return evaluate_single_factor(
                board, self.player, self.single_factor,
                game.get_legal_moves
            )
        else:
            return evaluate(board, self.player, game.get_legal_moves)

    def _minimax(self, board, depth, is_maximizing, game):
        """Pure Minimax without pruning."""
        if depth == 0 or game.is_game_over(board):
            return self._eval(board, game)

        current = self.player if is_maximizing else -self.player
        moves = game.get_legal_moves(board, current)

        if not moves:
            return self._minimax(board, depth - 1, not is_maximizing, game)

        if is_maximizing:
            best = float('-inf')
            for move in moves:
                new_board = game.make_move(board, move[0], move[1], current)
                best = max(best, self._minimax(new_board, depth - 1, False, game))
            return best
        else:
            best = float('inf')
            for move in moves:
                new_board = game.make_move(board, move[0], move[1], current)
                best = min(best, self._minimax(new_board, depth - 1, True, game))
            return best

    def _alphabeta(self, board, depth, alpha, beta, is_maximizing, game):
        """Alpha-Beta Pruning with move ordering."""
        if depth == 0 or game.is_game_over(board):
            return self._eval(board, game)

        current = self.player if is_maximizing else -self.player
        moves = game.get_legal_moves(board, current)

        if not moves:
            return self._alphabeta(board, depth - 1, alpha, beta,
                                   not is_maximizing, game)

        # Move ordering: sort moves by shallow evaluation (best first)
        ordered = self._order_moves(board, moves, current, game)

        if is_maximizing:
            best = float('-inf')
            for move in ordered:
                new_board = game.make_move(board, move[0], move[1], current)
                score = self._alphabeta(new_board, depth - 1, alpha, beta,
                                        False, game)
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break  # Prune
            return best
        else:
            best = float('inf')
            for move in ordered:
                new_board = game.make_move(board, move[0], move[1], current)
                score = self._alphabeta(new_board, depth - 1, alpha, beta,
                                        True, game)
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break  # Prune
            return best

    def _order_moves(self, board, moves, player, game):
        """
        Sort moves by shallow heuristic score — best first.
        This maximizes pruning efficiency in Alpha-Beta.
        """
        scored = []
        for move in moves:
            new_board = game.make_move(board, move[0], move[1], player)
            score = evaluate(new_board, self.player, game.get_legal_moves)
            scored.append((score, move))
        scored.sort(reverse=True)
        return [m for _, m in scored]

    def get_stats(self):
        """Return benchmarking stats as a dictionary."""
        return {
            'nodes_explored': self.nodes_explored,
            'time_seconds': round(self.last_move_time, 4),
            'depth': self.depth,
            'algorithm': 'Alpha-Beta' if self.use_alpha_beta else 'Minimax',
            'heuristic': self.heuristic_mode
        }
