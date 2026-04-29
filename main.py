"""
main.py
-------
Entry point for Othello AI.
Run modes:
  python main.py play          -- Human vs AI (Pygame GUI)
  python main.py ai            -- AI vs AI demo
  python main.py experiments   -- Run all experiments

Team Othello - ICSI435/535 Artificial Intelligence
University at Albany
"""

import sys

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "play"

    if mode == "play":
        print("Starting Human vs AI game...")
        print("NOTE: Pygame GUI requires display. Run on a machine with a screen.")
        try:
            from gui.othello_gui import run_game
            run_game()
        except ImportError:
            print("Pygame not installed. Run: pip install pygame")

    elif mode == "ai":
        print("Running AI vs AI demo (no GUI)...")
        from game.othello_game import OthelloGame, BLACK, WHITE
        from ai.ai_agent import OthelloAI

        game  = OthelloGame()
        black = OthelloAI(BLACK, depth=4, heuristic="multi")
        white = OthelloAI(WHITE, depth=4, heuristic="coin_parity")

        move_num = 0
        while not game.is_game_over():
            move_num += 1
            p = game.current_player
            ai = black if p == BLACK else white
            move = ai.get_best_move(game.board)
            if move:
                game.apply_move(*move)
                ai.print_stats()
            print(f"\nMove {move_num} ({'Black' if p==BLACK else 'White'}) → {move}")
            print(game)

        b, w = game.get_score()
        winner = game.get_winner()
        print(f"\n{'='*40}")
        print(f"Game Over! Black={b}  White={w}")
        print(f"Winner: {'Black (Multi-factor)' if winner==BLACK else 'White (Disc-count)' if winner==WHITE else 'Draw'}")

    elif mode == "experiments":
        print("Running all experiments...")
        from experiments.experiments import (
            experiment_node_count,
            experiment_heuristic_comparison,
            experiment_phase_aware,
            experiment_ablation,
            experiment_depth_vs_winrate
        )
        experiment_node_count()
        experiment_heuristic_comparison()
        experiment_phase_aware()
        experiment_ablation()
        experiment_depth_vs_winrate()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [play|ai|experiments]")


if __name__ == "__main__":
    main()
