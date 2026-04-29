"""
heuristics.py
Multi-Factor Heuristic Evaluation — Team Othello
ICSI435/535 Artificial Intelligence — University at Albany

Implements 5 heuristic factors with phase-aware dynamic weighting.
Based on Rosenbloom (1982) IAGO and Sannidhanam & Annamalai (2004).

Factors:
    1. Coin Parity    - disc count difference
    2. Mobility       - legal moves available
    3. Corner Control - corners captured
    4. Positional Weights - 8x8 strategic value matrix
    5. Stability      - discs that cannot be flipped
"""

import numpy as np
from othello_game import EMPTY, BLACK, WHITE, CORNERS, DIRECTIONS

# ── Positional weight matrix (Rosenbloom 1982 / Sannidhanam 2004) ──────────
WEIGHT_MATRIX = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [  10,  -2,   4,   2,   2,   4,  -2,   10],
    [   5,  -2,   2,   1,   1,   2,  -2,    5],
    [   5,  -2,   2,   1,   1,   2,  -2,    5],
    [  10,  -2,   4,   2,   2,   4,  -2,   10],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
], dtype=float)

# X-squares (diagonal to corners — dangerous)
X_SQUARES = [(1,1),(1,6),(6,1),(6,6)]
# C-squares (adjacent to corners on edges)
C_SQUARES = [(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)]


def get_phase(board):
    """
    Determine game phase based on number of discs on the board.
    Returns: 'early', 'mid', or 'late'
    """
    filled = int(np.count_nonzero(board))
    if filled < 20:
        return 'early'
    elif filled < 44:
        return 'mid'
    else:
        return 'late'


def get_phase_weights(phase):
    """
    Return heuristic weights for each game phase.
    [coin_parity, mobility, corners, positional, stability]

    Early: mobility is most important — keep options open
    Mid:   corners and stability dominate
    Late:  disc count (coin parity) becomes decisive
    """
    if phase == 'early':
        return {'coin_parity': 0.1, 'mobility': 1.0, 'corners': 0.8,
                'positional': 0.5, 'stability': 0.3}
    elif phase == 'mid':
        return {'coin_parity': 0.2, 'mobility': 0.6, 'corners': 1.0,
                'positional': 0.8, 'stability': 0.9}
    else:  # late
        return {'coin_parity': 1.0, 'mobility': 0.2, 'corners': 0.6,
                'positional': 0.3, 'stability': 0.8}


# ── Individual heuristic functions ──────────────────────────────────────────

def coin_parity(board, player):
    """
    Normalized disc count difference.
    Range: -100 to 100
    """
    p = int(np.sum(board == player))
    o = int(np.sum(board == -player))
    total = p + o
    if total == 0:
        return 0
    return 100 * (p - o) / total


def mobility(board, player, get_legal_moves_fn):
    """
    Actual mobility: normalized difference in legal moves available.
    Range: -100 to 100
    """
    p_moves = len(get_legal_moves_fn(board, player))
    o_moves = len(get_legal_moves_fn(board, -player))
    total = p_moves + o_moves
    if total == 0:
        return 0
    return 100 * (p_moves - o_moves) / total


def corner_control(board, player):
    """
    Corner occupancy difference.
    Corners can never be flipped — highest strategic value.
    Range: -100 to 100
    """
    p_corners = sum(1 for r, c in CORNERS if board[r][c] == player)
    o_corners = sum(1 for r, c in CORNERS if board[r][c] == -player)
    total = p_corners + o_corners
    if total == 0:
        return 0
    return 100 * (p_corners - o_corners) / total


def positional_weights(board, player):
    """
    Sum of strategic position values for each player's discs.
    Uses the 8x8 WEIGHT_MATRIX — corners high, X-squares negative.
    """
    p_score = float(np.sum(WEIGHT_MATRIX[board == player]))
    o_score = float(np.sum(WEIGHT_MATRIX[board == -player]))
    return p_score - o_score


def stability(board, player):
    """
    Simplified stability: counts discs that are stable
    (adjacent to a captured corner and expanding stably outward).
    Range: -100 to 100
    """
    def count_stable(p):
        stable = set()
        for cr, cc in CORNERS:
            if board[cr][cc] != p:
                continue
            # Expand from corner — discs in same row/col as a corner disc
            for dr in range(8):
                if board[dr][cc] == p:
                    stable.add((dr, cc))
                else:
                    break
            for dc in range(8):
                if board[cr][dc] == p:
                    stable.add((cr, dc))
                else:
                    break
        return len(stable)

    p_stable = count_stable(player)
    o_stable = count_stable(-player)
    total = p_stable + o_stable
    if total == 0:
        return 0
    return 100 * (p_stable - o_stable) / total


# ── Master evaluation function ───────────────────────────────────────────────

def evaluate(board, player, get_legal_moves_fn):
    """
    Combined phase-aware heuristic evaluation.

    Args:
        board: 8x8 numpy array
        player: BLACK (1) or WHITE (-1)
        get_legal_moves_fn: function from OthelloGame

    Returns:
        float: evaluation score (higher = better for player)
    """
    phase = get_phase(board)
    w = get_phase_weights(phase)

    score = 0.0
    score += w['coin_parity']  * coin_parity(board, player)
    score += w['mobility']     * mobility(board, player, get_legal_moves_fn)
    score += w['corners']      * corner_control(board, player)
    score += w['positional']   * positional_weights(board, player)
    score += w['stability']    * stability(board, player)

    return score


def evaluate_single_factor(board, player, factor, get_legal_moves_fn):
    """
    Evaluate using only one heuristic factor.
    Used for ablation experiments.

    Args:
        factor: 'coin_parity' | 'mobility' | 'corners' | 'positional' | 'stability'
    """
    if factor == 'coin_parity':
        return coin_parity(board, player)
    elif factor == 'mobility':
        return mobility(board, player, get_legal_moves_fn)
    elif factor == 'corners':
        return corner_control(board, player)
    elif factor == 'positional':
        return positional_weights(board, player)
    elif factor == 'stability':
        return stability(board, player)
    return 0.0
