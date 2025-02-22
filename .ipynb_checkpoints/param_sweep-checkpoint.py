# param_sweep.py

import config
from main import run_main_simulation
import csv
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parameter_sweep_basic_strategy_averaged():
    """
    FINAL approach for parameter sweeping in Basic Strategy, with averaging:
    
      - deck_counts = [1, 2, 3, 4, 6, 8]
      - shuffle_points = [0.25, 0.35, 0.50, 0.75]
      - repeats = 15 (number of repeated runs per combination)
      - shoes_per_run = 500
      - average the metrics across the repeats
    
    It produces:
      1. A CSV ("param_sweep_averaged.csv") with one row per (deck_count, shuffle_point),
         including averaged EV/hand, net profit, win/loss/push rates, etc.
      2. A heatmap of average EV/hand ("static/param_sweep_averaged_heatmap.png")
      3. A line plot of average EV% vs. # of decks ("static/param_sweep_averaged_lineplot.png")
    """

    # 1) Parameters for the sweep
    deck_counts    = [1, 2, 3, 4, 6, 8]
    shuffle_points = [0.25, 0.33, 0.50, 0.75]
    repeats        = 15
    shoes_per_run  = 500

    # 2) Configure Basic Strategy with a flat $10 bet
    config.RL_METHOD     = "BasicStrategy"
    config.BETTING_STYLE = "flat"
    config.DEFAULT_WAGER = 10  # used to compute EV% = (EV_per_hand / wager) * 100

    # We'll store all results in a list of dicts
    final_results = []

    # 3) Nested loop: deck_counts x shuffle_points
    for d in deck_counts:
        for sp in shuffle_points:
            # Prepare lists to accumulate metrics across the repeats
            ev_list         = []
            net_profit_list = []
            win_rate_list   = []
            loss_rate_list  = []
            push_rate_list  = []

            # Set deck config
            config.NUM_DECKS   = d
            config.TOTAL_CARDS = 52 * d

            # Run 'repeats' times for each combo
            for _ in range(repeats):
                config.SHUFFLE_POINT = sp

                # Run the simulation
                outcome, agent, logger = run_main_simulation(
                    agent_override=None,
                    num_shoes=shoes_per_run
                )
                summary = outcome["summary"]

                # Gather stats for this run
                ev_list.append(summary["EV_per_hand"])
                net_profit_list.append(summary["net_profit"])
                win_rate_list.append(summary["win_rate"])
                loss_rate_list.append(summary["loss_rate"])
                push_rate_list.append(summary["push_rate"])

            # Compute averages across the repeats
            avg_ev       = statistics.mean(ev_list)
            avg_profit   = statistics.mean(net_profit_list)
            avg_win_rate = statistics.mean(win_rate_list)
            avg_loss_rate= statistics.mean(loss_rate_list)
            avg_push_rate= statistics.mean(push_rate_list)

            ev_percent = (avg_ev / config.DEFAULT_WAGER) * 100

            row = {
                "num_decks": d,
                "shuffle_point": sp,
                "repeats": repeats,
                "shoes_per_run": shoes_per_run,
                "avg_EV_per_hand": avg_ev,
                "avg_EV_percent": ev_percent,
                "avg_net_profit": avg_profit,
                "avg_win_rate": avg_win_rate,
                "avg_loss_rate": avg_loss_rate,
                "avg_push_rate": avg_push_rate
            }
            final_results.append(row)

    # 4) Write aggregated results to CSV
    csv_file = "param_sweep_averaged.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
        writer.writeheader()
        writer.writerows(final_results)

    # Convert to DataFrame
    df = pd.DataFrame(final_results)

    # Optional: Set a nicer Seaborn style
    sns.set(style="whitegrid", font_scale=1.1)

    # ---------- (A) HEATMAP of avg_EV_per_hand ----------
    pivot_ev = df.pivot(index="num_decks", columns="shuffle_point", values="avg_EV_per_hand")

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        pivot_ev, annot=True, cmap="viridis", fmt=".4g",
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Average EV/Hand (Basic Strategy)\n(Decks vs. Shuffle Point), Averaged Over 15x500 Shoes")
    plt.xlabel("Shuffle Point")
    plt.ylabel("Number of Decks")
    heatmap_file = "static/param_sweep_averaged_heatmap.png"
    plt.savefig(heatmap_file, bbox_inches="tight")
    plt.close()

    # ---------- (B) LINE PLOT of avg_EV_percent vs. # Decks ----------
    line_fig, ax = plt.subplots(figsize=(9,6))
    for sp, subset in df.groupby("shuffle_point"):
        subset = subset.sort_values(by="num_decks")
        ax.plot(
            subset["num_decks"],
            subset["avg_EV_percent"],
            marker='o',
            label=f"Shuffle Point: {sp}"
        )
    ax.set_xlabel("Number of Decks")
    ax.set_ylabel("Average EV% (relative to $10 bet)")
    ax.set_title("Average EV% by # of Decks\n(One line per Shuffle Point)")
    ax.legend(title="Shuffle Point")
    lineplot_file = "static/param_sweep_averaged_lineplot.png"
    line_fig.savefig(lineplot_file, bbox_inches="tight")
    plt.close(line_fig)

    print("\n====================== DONE ======================")
    print(f"CSV of averaged results: {csv_file}")
    print(f"Heatmap of EV/hand: {heatmap_file}")
    print(f"Line plot of EV%: {lineplot_file}\n")


# Uncomment the block below if you want to run this directly from command line:
if __name__ == "__main__":
    parameter_sweep_basic_strategy_averaged()