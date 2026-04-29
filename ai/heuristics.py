"""
heuristics.py
-------------
Multi-factor heuristic evaluation for Othello AI.

Implements 5 classical heuristic factors:
  1. Coin Parity    - disc count difference
  2. Mobility       - actual + potential legal moves
  3. Corner Control - corners captured (permanent, unflippable)
  4. Positional Weights - 8x8 strategic value matrix
  5. Stability      - discs that can never be flipped

Phase-aware weighting (our original contribution):
  Early game  (< 20 filled): mobility-first
  Mid game    (20-44 filled): corners + stability
  End game    (> 44 filled):  disc parity decisive

References:
  Rosenbloom (1982) - IAGO world-championship heuristics
  Sannidhanam & Annamalai - An Analysis of Heuristics in Othello

Team Othello - ICSI435/535 Artificial Intelligence
University at Albany
"""

import numpy as np

BLACK =  1
WHITE = -1
EMPTY =  0

CORNERS = [(0,0),(0,7),(7,0),(7,7)]

# X-squares (diagonal to corners — dangerous to play into)
X_SQUARES = [(1,1),(1,6),(6,1),(6,6)]

# C-squares (adjacent to corners on edges)
C_SQUARES = [(0,1),(1,0),(0,6),(1,7),(6,0),(7,1),(6,7),(7,6)]

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ── Positional weight matrix (Sannidhanam 2005) ───────────────
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


# ── Phase-aware weight configurations ─────────────────────────
PHASE_WEIGHTS = {
    "early": {
        "coin_parity":  0.1,
        "mobility":     1.0,
        "corners":      0.8,
        "positional":   0.5,
        "stability":    0.2,
    },
    "mid": {
        "coin_parity":  0.2,
        "mobility":     0.5,
        "corners":      1.0,
        "positional":   0.8,
        "stability":    0.9,
    },
    "end": {
        "coin_parity":  1.0,
        "mobility":     0.2,
        "corners":      0.5,
        "positional":   0.3,
        "stability":    0.8,
    },
}


def get_phase(board):
    filled = int(np.count_nonzero(board))
    if filled < 20:
        return "early"
    elif filled < 44:
        return "mid"
    return "end"


# ── Individual heuristic functions ────────────────────────────

def coin_parity(board, player):
    """
    Normalized disc count difference.
    Range: -100 to +100
    Low weight early, high weight endgame.
    """
    p = int(np.sum(board == player))
    o = int(np.sum(board == -player))
    total = p + o
    if total == 0:
        return 0
    return 100 * (p - o) / total


def get_legal_moves_for(board, player):
    """Helper: get legal moves for any player on any board."""
    opponent = -player
    legal = []
    for r in range(8):
        for c in range(8):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                found = False
                while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                    nr += dr; nc += dc
                    found = True
                if found and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                    legal.append((r, c))
                    break
    return legal


def mobility(board, player):
    """
    Actual mobility: legal moves available to each player.
    Restricting opponent mobility is often more valuable than gaining discs.
    Range: -100 to +100
    """
    p_moves = len(get_legal_moves_for(board, player))
    o_moves = len(get_legal_moves_for(board, -player))
    total = p_moves + o_moves
    if total == 0:
        return 0
    return 100 * (p_moves - o_moves) / total


def potential_mobility(board, player):
    """
    Potential mobility: empty squares adjacent to opponent discs.
    Forward-looking measure of future move availability.
    """
    opponent = -player
    p_potential = set()
    o_potential = set()
    for r in range(8):
        for c in range(8):
            if board[r][c] == opponent:
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == EMPTY:
                        p_potential.add((nr, nc))
            elif board[r][c] == player:
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == EMPTY:
                        o_potential.add((nr, nc))
    p = len(p_potential)
    o = len(o_potential)
    if p + o == 0:
        return 0
    return 100 * (p - o) / (p + o)


def corner_control(board, player):
    """
    Corners captured. Corners can never be flipped — permanent anchors.
    Always heavily weighted regardless of game phase.
    Range: -100 to +100
    """
    p_corners = sum(1 for r, c in CORNERS if board[r][c] == player)
    o_corners = sum(1 for r, c in CORNERS if board[r][c] == -player)
    total = p_corners + o_corners
    if total == 0:
        return 0
    return 100 * (p_corners - o_corners) / total


def corner_closeness(board, player):
    """
    Penalty for occupying X-squares and C-squares when corners are empty.
    Playing into X-squares hands corners to the opponent.
    """
    opponent = -player
    p_penalty = 0
    o_penalty = 0
    for (cr, cc), (xr, xc) in zip(CORNERS, X_SQUARES):
        if board[cr][cc] == EMPTY:
            if board[xr][xc] == player:
                p_penalty += 1
            elif board[xr][xc] == opponent:
                o_penalty += 1
    return -12.5 * (p_penalty - o_penalty)


def positional_weight(board, player):
    """
    8x8 positional weight matrix — strategic value of each square.
    Corners=100 (permanent), X-squares=-50 (dangerous), edges=10.
    """
    p_score = float(np.sum(WEIGHT_MATRIX[board == player]))
    o_score = float(np.sum(WEIGHT_MATRIX[board == -player]))
    return p_score - o_score


def stability(board, player):
    """
    Stability: discs that cannot be flipped for the rest of the game.
    Simplified version: corners + edges adjacent to captured corners count as stable.
    Range: -100 to +100
    """
    def count_stable(p):
        stable = set()
        # Corners are always stable
        for r, c in CORNERS:
            if board[r][c] == p:
                stable.add((r, c))
                # Spread stability along edges from corners
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == p:
                        stable.add((nr, nc))
                        nr += dr
                        nc += dc
        return len(stable)

    p_stable = count_stable(player)
    o_stable = count_stable(-player)
    total = p_stable + o_stable
    if total == 0:
        return 0
    return 100 * (p_stable - o_stable) / total


# ── Combined evaluation function ──────────────────────────────

def evaluate(board, player, phase=None, custom_weights=None):
    """
    Combined multi-factor heuristic evaluation.

    Args:
        board:          8x8 numpy array
        player:         1 (Black) or -1 (White)
        phase:          'early', 'mid', 'end' — auto-detected if None
        custom_weights: dict override for weights

    Returns:
        float: evaluation score (positive = good for player)
    """
    if phase is None:
        phase = get_phase(board)

    w = custom_weights if custom_weights else PHASE_WEIGHTS[phase]

    score = (
        w["coin_parity"]  * coin_parity(board, player)       +
        w["mobility"]     * mobility(board, player)           +
        w["corners"]      * corner_control(board, player)     +
        w["positional"]   * positional_weight(board, player)  +
        w["stability"]    * stability(board, player)
    )
    return score


def evaluate_single_factor(board, player, factor):
    """
    Evaluate using only one factor — used in ablation experiments.
    factor: 'coin_parity' | 'mobility' | 'corners' | 'positional' | 'stability'
    """
    factor_map = {
        "coin_parity": coin_parity,
        "mobility":    mobility,
        "corners":     corner_control,
        "positional":  positional_weight,
        "stability":   stability,
    }
    if factor not in factor_map:
        raise ValueError(f"Unknown factor: {factor}")
    return factor_map[factor](board, player)
