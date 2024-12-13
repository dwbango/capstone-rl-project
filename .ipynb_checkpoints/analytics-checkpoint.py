# analytics 
import matplotlib.pyplot as plt
import statistics
import config

class DataLogger:
    def __init__(self):
        # Each record will store details about a single hand
        # For example: {"hand_number": int, "outcome": str, "profit": float, "bankroll": float}
        self.records = []
        self.hand_counter = 0

    def log_hand(self, outcome, profit, bankroll):
        # outcome is 'win', 'lose', or 'push'
        # profit is the net amount won or lost this hand
        # bankroll is the player's bankroll after this hand
        self.records.append({
            "hand_number": self.hand_counter,
            "outcome": outcome,
            "profit": profit,
            "bankroll": bankroll
        })
        self.hand_counter += 1

    def get_data(self):
        # Return a copy of the records as a list of dicts
        return self.records[:]

    def get_outcomes(self):
        # Return a list of outcomes for all hands
        return [r["outcome"] for r in self.records]

    def get_profits(self):
        # Return a list of profits per hand
        return [r["profit"] for r in self.records]

    def get_bankroll_history(self):
        # Return bankroll after each hand
        return [r["bankroll"] for r in self.records]

    def get_counts(self):
        # Count wins, losses, pushes
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
    plt.show()

def compute_ev_per_hand(profits):
    # EV = average profit per hand
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
    # Print a summary of results using the logger data
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

    print("\n--- Simulation Summary ---")
    print(f"Total Hands Played: {total_hands}")
    print(f"Wins: {wins} ({win_rate*100:.2f}%)")
    print(f"Losses: {losses} ({loss_rate*100:.2f}%)")
    print(f"Pushes: {pushes} ({push_rate*100:.2f}%)")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"Net Profit/Loss: ${net_profit:.2f}")
    print(f"EV per hand: ${ev:.2f} ({ev_pct:.3f}%)")
    print(f"Variance in profits: {var:.2f}")