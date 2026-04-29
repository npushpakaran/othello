"""
experiments.py
--------------
Heuristic comparison experiments for Othello AI.

Experiments:
  1. Node count: Minimax vs Alpha-Beta at depths 3-6
  2. Single factor vs combined 5-factor heuristic (20 games each)
  3. Static weights vs phase-aware weights (20 games)
  4. Ablation: remove one factor at a time
  5. Depth vs win rate vs random agent

Team Othello - ICSI435/535 Artificial Intelligence
University at Albany
"""

import sys, os, time, random, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from game.othello_game import OthelloGame, BLACK, WHITE
from ai.ai_agent import OthelloAI, make_move_on_board
from ai.heuristics import get_legal_moves_for, PHASE_WEIGHTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────

def random_move(board, player):
    """Pick a random legal move."""
    moves = get_legal_moves_for(board, player)
    return random.choice(moves) if moves else None


def play_game(black_ai=None, white_ai=None, verbose=False):
    """
    Play one game. AI=None means random agent.
    Returns: (winner, black_score, white_score, total_moves)
    """
    game = OthelloGame()
    while not game.is_game_over():
        p = game.current_player
        if p == BLACK:
            move = black_ai.get_best_move(game.board) if black_ai else random_move(game.board, BLACK)
        else:
            move = white_ai.get_best_move(game.board) if white_ai else random_move(game.board, WHITE)

        if move is None:
            game._next_turn()
            continue
        game.apply_move(*move)

    b, w = game.get_score()
    winner = game.get_winner()
    if verbose:
        print(f"  Result: Black={b} White={w} Winner={'Black' if winner==BLACK else 'White' if winner==WHITE else 'Draw'}")
    return winner, b, w, game.move_count


def win_rate(results, player):
    wins = sum(1 for w, _, _, _ in results if w == player)
    return round(100 * wins / len(results), 1)


# ── Experiment 1: Node count Minimax vs Alpha-Beta ────────────

def experiment_node_count(depths=[3, 4, 5, 6], trials=5):
    """Measure nodes explored at each depth with vs without pruning."""
    print("\n=== Experiment 1: Node Count - Minimax vs Alpha-Beta ===")
    minimax_nodes = []
    ab_nodes      = []

    for depth in depths:
        m_total = 0
        a_total = 0
        m_time  = 0
        a_time  = 0

        for _ in range(trials):
            game = OthelloGame()
            # Advance to mid-game position
            for _ in range(10):
                moves = game.get_legal_moves()
                if moves and not game.is_game_over():
                    game.apply_move(*random.choice(moves))

            # Minimax (no pruning — simulate by setting alpha/beta to fixed values)
            ai_mm = OthelloAI(game.current_player, depth=depth, heuristic="multi")
            # For pure minimax comparison, use very wide alpha-beta (effectively no pruning)
            ai_mm.reset_counters()
            t0 = time.time()
            ai_mm._alphabeta(game.board, depth, -99999, 99999, True, game.current_player)
            m_time  += time.time() - t0
            m_total += ai_mm.nodes_explored

            # Alpha-Beta with move ordering
            ai_ab = OthelloAI(game.current_player, depth=depth, heuristic="multi")
            ai_ab.reset_counters()
            t0 = time.time()
            ai_ab.get_best_move(game.board)
            a_time  += time.time() - t0
            a_total += ai_ab.nodes_explored

        mm_avg = m_total // trials
        ab_avg = a_total // trials
        reduction = round(100 * (mm_avg - ab_avg) / mm_avg, 1) if mm_avg > 0 else 0

        minimax_nodes.append(mm_avg)
        ab_nodes.append(ab_avg)
        print(f"  Depth {depth}: Minimax={mm_avg:,}  Alpha-Beta={ab_avg:,}  "
              f"Reduction={reduction}%  "
              f"MM_time={m_time/trials:.3f}s  AB_time={a_time/trials:.3f}s")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 1: Node Count — Minimax vs Alpha-Beta Pruning", fontsize=13, fontweight='bold')

    x = list(depths)
    ax1.plot(x, minimax_nodes, 'o-', color='#C0392B', linewidth=2.5, markersize=7, label='Minimax')
    ax1.plot(x, ab_nodes,      's--', color='#1D9E75', linewidth=2.5, markersize=7, label='Alpha-Beta')
    ax1.set_yscale('log')
    ax1.set_xlabel('Search Depth')
    ax1.set_ylabel('Nodes Explored (log scale)')
    ax1.set_title('Nodes Explored per Move')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)

    reductions = [round(100*(m-a)/m, 1) if m>0 else 0 for m,a in zip(minimax_nodes, ab_nodes)]
    bars = ax2.bar([f'Depth {d}' for d in depths], reductions,
                   color=['#2E9E85','#2E9E85','#2E9E85','#1D9E75'], edgecolor='white')
    for bar, val in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Node Reduction (%)')
    ax2.set_title('Pruning Efficiency')
    ax2.set_ylim(0, 70)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp1_node_count.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return {"minimax_nodes": minimax_nodes, "ab_nodes": ab_nodes, "depths": depths}


# ── Experiment 2: Single factor vs combined heuristic ─────────

def experiment_heuristic_comparison(n_games=20, depth=4):
    """Compare disc-count AI vs 5-factor AI (20 games each matchup)."""
    print(f"\n=== Experiment 2: Single Factor vs Combined ({n_games} games each) ===")
    factors = ["coin_parity", "mobility", "corners", "positional", "stability"]
    win_rates = {}

    for factor in factors:
        results = []
        for i in range(n_games):
            # Factor AI plays Black, Multi AI plays White
            black_ai = OthelloAI(BLACK, depth=depth, heuristic=factor)
            white_ai = OthelloAI(WHITE, depth=depth, heuristic="multi")
            winner, b, w, _ = play_game(black_ai, white_ai)
            results.append((winner, b, w, 0))
            if (i+1) % 5 == 0:
                print(f"  {factor}: {i+1}/{n_games} games done")

        single_wr = win_rate(results, BLACK)
        multi_wr  = win_rate(results, WHITE)
        win_rates[factor] = {"single": single_wr, "multi": multi_wr}
        print(f"  {factor:15s}: Single={single_wr}%  Multi={multi_wr}%  "
              f"Multi wins more: {multi_wr > single_wr}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Experiment 2: Single Factor vs Combined 5-Factor Heuristic", fontsize=13, fontweight='bold')

    x = np.arange(len(factors))
    w = 0.35
    single_vals = [win_rates[f]["single"] for f in factors]
    multi_vals  = [win_rates[f]["multi"]  for f in factors]

    bars1 = ax.bar(x - w/2, single_vals, w, label='Single factor (Black)', color='#E67E22', alpha=0.85)
    bars2 = ax.bar(x + w/2, multi_vals,  w, label='5-factor combined (White)', color='#1D9E75', alpha=0.85)

    for bar, val in zip(bars1, single_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, multi_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_',' ').title() for f in factors])
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp2_heuristic_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return win_rates


# ── Experiment 3: Static vs Phase-aware weights ───────────────

def experiment_phase_aware(n_games=20, depth=4):
    """Compare static weights vs phase-aware dynamic weights."""
    print(f"\n=== Experiment 3: Static vs Phase-Aware Weights ({n_games} games) ===")

    # Static = mid-game weights fixed for entire game
    static_weights = PHASE_WEIGHTS["mid"]

    results = []
    for i in range(n_games):
        # Static plays Black, Phase-aware plays White
        black_ai = OthelloAI(BLACK, depth=depth, heuristic="multi", custom_weights=static_weights)
        white_ai = OthelloAI(WHITE, depth=depth, heuristic="multi")  # phase-aware by default
        winner, b, w, _ = play_game(black_ai, white_ai)
        results.append((winner, b, w, 0))
        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_games} games done")

    static_wr = win_rate(results, BLACK)
    phase_wr  = win_rate(results, WHITE)
    draws     = round(100 - static_wr - phase_wr, 1)
    print(f"  Static weights: {static_wr}%   Phase-aware: {phase_wr}%   Draws: {draws}%")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Experiment 3: Static vs Phase-Aware Weights", fontsize=13, fontweight='bold')
    categories = ['Static Weights\n(fixed mid-game)', 'Phase-Aware\n(dynamic)', 'Draws']
    values = [static_wr, phase_wr, draws]
    colors = ['#E67E22', '#1D9E75', '#95A5A6']
    bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp3_phase_aware.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return {"static": static_wr, "phase_aware": phase_wr, "draws": draws}


# ── Experiment 4: Ablation — remove one factor at a time ──────

def experiment_ablation(n_games=15, depth=4):
    """
    Ablation study: remove each heuristic factor one at a time.
    The factor whose removal hurts win rate the most is the most important.
    """
    print(f"\n=== Experiment 4: Ablation Study ({n_games} games each) ===")
    factors = ["coin_parity", "mobility", "corners", "positional", "stability"]
    ablation_results = {}

    # Baseline: all factors
    baseline = []
    for _ in range(n_games):
        ai_b = OthelloAI(BLACK, depth=depth, heuristic="multi")
        winner, b, w, _ = play_game(ai_b, None)  # vs random
        baseline.append((winner, b, w, 0))
    baseline_wr = win_rate(baseline, BLACK)
    print(f"  Baseline (all factors) vs random: {baseline_wr}%")

    for factor in factors:
        # Build weights with this factor zeroed out
        ablated = {k: v for k, v in PHASE_WEIGHTS["mid"].items()}
        ablated[factor] = 0.0

        results = []
        for _ in range(n_games):
            ai_b = OthelloAI(BLACK, depth=depth, heuristic="multi", custom_weights=ablated)
            winner, b, w, _ = play_game(ai_b, None)
            results.append((winner, b, w, 0))

        wr = win_rate(results, BLACK)
        drop = round(baseline_wr - wr, 1)
        ablation_results[factor] = {"win_rate": wr, "drop": drop}
        print(f"  Without {factor:15s}: {wr}%  (drop: -{drop}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Experiment 4: Ablation Study — Impact of Each Heuristic Factor", fontsize=13, fontweight='bold')
    factor_names = [f.replace('_',' ').title() for f in factors]
    drops = [ablation_results[f]["drop"] for f in factors]
    colors = ['#C0392B' if d == max(drops) else '#E67E22' if d > 5 else '#3498DB' for d in drops]
    bars = ax.bar(factor_names, drops, color=colors, edgecolor='white')
    for bar, val in zip(bars, drops):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'-{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Win Rate Drop When Factor Removed (%)')
    ax.set_title('Larger drop = more important factor')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp4_ablation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return ablation_results


# ── Experiment 5: Depth vs win rate ───────────────────────────

def experiment_depth_vs_winrate(depths=[2,3,4,5], n_games=15):
    """Win rate vs random agent at different search depths."""
    print(f"\n=== Experiment 5: Depth vs Win Rate ({n_games} games per depth) ===")
    results_by_depth = {}
    times_by_depth   = {}

    for depth in depths:
        wins = 0
        total_time = 0
        for i in range(n_games):
            ai_b = OthelloAI(BLACK, depth=depth, heuristic="multi")
            winner, b, w, _ = play_game(ai_b, None)
            if winner == BLACK:
                wins += 1
            total_time += ai_b.last_move_time
            if (i+1) % 5 == 0:
                print(f"  Depth {depth}: {i+1}/{n_games} done")

        wr = round(100 * wins / n_games, 1)
        avg_t = round(total_time / n_games, 3)
        results_by_depth[depth] = wr
        times_by_depth[depth]   = avg_t
        print(f"  Depth {depth}: Win rate={wr}%  Avg time={avg_t}s")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle("Experiment 5: Search Depth vs Win Rate vs Time", fontsize=13, fontweight='bold')
    ax2 = ax1.twinx()

    x = list(depths)
    wr_vals = [results_by_depth[d] for d in x]
    t_vals  = [times_by_depth[d]   for d in x]

    ax1.plot(x, wr_vals, 'o-', color='#1D9E75', linewidth=2.5, markersize=8, label='Win rate (%)')
    ax1.set_xlabel('Search Depth')
    ax1.set_ylabel('Win Rate vs Random Agent (%)', color='#1D9E75')
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', labelcolor='#1D9E75')
    ax1.set_xticks(x)

    ax2.plot(x, t_vals, 's--', color='#C0392B', linewidth=2, markersize=7, label='Avg time (s)')
    ax2.set_ylabel('Average Move Time (seconds)', color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'exp5_depth_winrate.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return {"win_rates": results_by_depth, "times": times_by_depth}


# ── Run all experiments ───────────────────────────────────────

if __name__ == "__main__":
    print("Running all Othello AI experiments...")
    print("This will take several minutes.\n")

    all_results = {}
    all_results["exp1"] = experiment_node_count(depths=[3,4,5,6], trials=3)
    all_results["exp2"] = experiment_heuristic_comparison(n_games=20, depth=4)
    all_results["exp3"] = experiment_phase_aware(n_games=20, depth=4)
    all_results["exp4"] = experiment_ablation(n_games=15, depth=4)
    all_results["exp5"] = experiment_depth_vs_winrate(depths=[2,3,4,5], n_games=15)

    # Save results as JSON
    results_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ All experiments complete. Results saved to {RESULTS_DIR}/")
    print("Graphs saved as PNG files. JSON summary saved as all_results.json")
