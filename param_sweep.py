# param_sweep.py
import config
from main import run_main_simulation

import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parameter_sweep_basic_strategy():
    """
    Demonstration of a parameter sweep for Basic Strategy:
      - Varies the number of decks
      - Varies the shuffle penetration
      - Runs run_main_simulation() for each combination
      - Collects EV/Hand and other stats in a CSV
      - Generates a heatmap to visualize the EV changes
    """

    # 1) Define the parameter ranges and # of shoes per run
    deck_counts      = [1, 2, 4, 6, 8]      # for example
    shuffle_points   = [0.25, 0.33, 0.50]   # 25%, 33%, 50% penetration
    num_shoes_each   = 5000                # how many shoes to simulate for each combo

    # 2) Configure the basics for Basic Strategy
    config.RL_METHOD     = "BasicStrategy"
    config.BETTING_STYLE = "flat"
    config.DEFAULT_WAGER = 10

    # We'll store the results in a list of dictionaries
    results = []

    # 3) Loop over all parameter combos
    for d in deck_counts:
        for sp in shuffle_points:
            # Update config for each combo
            config.NUM_DECKS     = d
            config.TOTAL_CARDS   = 52 * d
            config.SHUFFLE_POINT = sp

            # Call run_main_simulation, overriding the # of shoes each time
            outcome, agent, logger = run_main_simulation(
                agent_override=None,
                num_shoes=num_shoes_each
            )

            summary = outcome["summary"]

            # Store the key metrics
            row = {
                "num_decks": d,
                "shuffle_point": sp,
                "hands_played": outcome["hands_played"],
                "shoes_played": outcome["shoes_played"],
                "EV_per_hand": summary["EV_per_hand"],
                "net_profit": summary["net_profit"],
                "win_rate": summary["win_rate"],
                "loss_rate": summary["loss_rate"],
                "push_rate": summary["push_rate"]
            }
            results.append(row)

    # 4) Write a CSV with all results
    csv_file = "param_sweep_results.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 5) Create a heatmap of EV per hand
    #    - We'll pivot the data so "num_decks" is on one axis, "shuffle_point" on the other
    df = pd.DataFrame(results)
    pivot_df = df.pivot(index="num_decks", columns="shuffle_point", values="EV_per_hand")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap="viridis")
    plt.title("EV per Hand (Basic Strategy) across #Decks vs. Shuffle Point")
    plt.xlabel("Shuffle Penetration")
    plt.ylabel("Number of Decks")

    # Save plot to your static directory so you can view it from the app if desired
    heatmap_file = "static/param_sweep_heatmap.png"
    plt.savefig(heatmap_file)
    plt.close()

    print(f"Done!\n- Wrote results to {csv_file}\n- Heatmap saved to {heatmap_file}")

if __name__ == "__main__":
    parameter_sweep_basic_strategy()