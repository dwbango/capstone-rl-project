# analytics.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import statistics
import config

class DataLogger:
    def __init__(self):
        self.records = []
        self.hand_counter = 0

    def log_hand(self, outcome, profit, bankroll):
        self.records.append({
            "hand_number": self.hand_counter,
            "outcome": outcome,
            "profit": profit,
            "bankroll": bankroll
        })
        self.hand_counter += 1

    def get_data(self):
        return self.records[:]

    def get_outcomes(self):
        return [r["outcome"] for r in self.records]

    def get_profits(self):
        return [r["profit"] for r in self.records]

    def get_bankroll_history(self):
        return [r["bankroll"] for r in self.records]

    def get_counts(self):
        wins = sum(1 for r in self.records if r["outcome"] == "win")
        losses = sum(1 for r in self.records if r["outcome"] == "lose")
        pushes = sum(1 for r in self.records if r["outcome"] == "push")
        return wins, losses, pushes

def plot_bankroll_over_time(bankroll_history):
    plt.figure(figsize=(10,5))
    plt.plot(range(len(bankroll_history)), bankroll_history, label='Bankroll')
    plt.title("Bankroll Over Time")
    plt.xlabel("Hand Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.legend()
    plt.savefig('static/bankroll_history.png')  # Save plot to static folder
    plt.close()

def compute_ev_per_hand(profits):
    if not profits:
        return 0.0
    return sum(profits) / len(profits)

def compute_outcome_rates(wins, losses, pushes, total_hands):
    if total_hands == 0:
        return 0.0, 0.0, 0.0
    win_rate = wins / total_hands
    loss_rate = losses / total_hands
    push_rate = pushes / total_hands
    return win_rate, loss_rate, push_rate

def compute_variance(profits):
    if len(profits) <= 1:
        return 0.0
    return statistics.pvariance(profits)

def print_summary(logger: DataLogger):
    records = logger.get_data()
    total_hands = len(records)
    wins, losses, pushes = logger.get_counts()
    profits = logger.get_profits()
    ev = compute_ev_per_hand(profits)
    var = compute_variance(profits)
    win_rate, loss_rate, push_rate = compute_outcome_rates(wins, losses, pushes, total_hands)
    final_bankroll = logger.get_bankroll_history()[-1] if total_hands > 0 else config.STARTING_BANKROLL
    net_profit = final_bankroll - config.STARTING_BANKROLL
    wager = config.DEFAULT_WAGER
    ev_pct = (ev / wager) if wager != 0 else 0.0

    summary = {
        "total_hands": total_hands,
        "wins": wins,
        "win_rate": win_rate,
        "losses": losses,
        "loss_rate": loss_rate,
        "pushes": pushes,
        "push_rate": push_rate,
        "final_bankroll": final_bankroll,
        "net_profit": net_profit,
        "EV_per_hand": ev,
        "EV_percent": ev_pct,
        "variance": var
    }

    if config.verbose:
        config.log_message("\n--- Simulation Summary ---")
        config.log_message(f"Total Hands Played: {total_hands}")
        config.log_message(f"Wins: {wins} ({win_rate*100:.2f}%)")
        config.log_message(f"Losses: {losses} ({loss_rate*100:.2f}%)")
        config.log_message(f"Pushes: {pushes} ({push_rate*100:.2f}%)")
        config.log_message(f"Final Bankroll: ${final_bankroll:.2f}")
        config.log_message(f"Net Profit/Loss: ${net_profit:.2f}")
        config.log_message(f"EV per hand: ${ev:.2f} ({ev_pct:.3f}%)")
        config.log_message(f"Variance in profits: {var:.2f}")

    return summary
