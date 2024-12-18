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
        self.shoe_records = []  # NEW: to store card orders per shoe

    def log_hand(self, outcome, profit, bankroll, actions_taken, starting_true_count, starting_decks_remaining, dealer_actions=[]):
        self.records.append({
            "hand_number": self.hand_counter,
            "outcome": outcome,
            "profit": profit,
            "bankroll": bankroll,
            "actions_taken": actions_taken,
            "dealer_actions": dealer_actions,  # NEW: log dealer actions
            "starting_true_count": starting_true_count,
            "starting_decks_remaining": starting_decks_remaining
        })
        self.hand_counter += 1

    def log_shoe_data(self, shoe_number, card_order):
        # NEW: Log the entire order of cards dealt this shoe
        self.shoe_records.append({
            "shoe_number": shoe_number,
            "card_order": card_order
        })

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
    plt.savefig('static/bankroll_history.png')
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

def tc_to_bin(tc):
    if tc <= -7:
        return "≤ -7"
    elif tc >= 7:
        return "≥ 7"
    else:
        return str(tc)

def print_summary(logger: DataLogger):
    records = logger.get_data()
    total_hands = len(records)
    wins, losses, pushes = logger.get_counts()
    profits = [r["profit"] for r in records]
    ev = compute_ev_per_hand(profits)
    var = compute_variance(profits)
    win_rate, loss_rate, push_rate = compute_outcome_rates(wins, losses, pushes, total_hands)
    final_bankroll = records[-1]["bankroll"] if total_hands > 0 else config.STARTING_BANKROLL
    net_profit = final_bankroll - config.STARTING_BANKROLL
    wager = config.DEFAULT_WAGER
    ev_pct = (ev / wager) if wager != 0 else 0.0

    action_stats = {}
    for r in records:
        outcome = r["outcome"]
        for a in r.get("actions_taken", []):
            if a not in action_stats:
                action_stats[a] = {"win":0, "lose":0, "push":0, "total":0}
            action_stats[a][outcome] += 1
            action_stats[a]["total"] += 1

    for a, stats in action_stats.items():
        stats["win_rate"] = stats["win"] / stats["total"] if stats["total"] > 0 else 0
        stats["loss_rate"] = stats["lose"] / stats["total"] if stats["total"] > 0 else 0
        stats["push_rate"] = stats["push"] / stats["total"] if stats["total"] > 0 else 0

    tc_bins = {}
    for r in records:
        tc = r.get("starting_true_count", 0)
        bin_label = tc_to_bin(tc)
        outcome = r["outcome"]
        if bin_label not in tc_bins:
            tc_bins[bin_label] = {"win":0, "lose":0, "push":0, "total":0}
        tc_bins[bin_label][outcome] += 1
        tc_bins[bin_label]["total"] += 1

    for b, stats in tc_bins.items():
        stats["win_rate"] = stats["win"]/stats["total"] if stats["total"] > 0 else 0
        stats["loss_rate"] = stats["lose"]/stats["total"] if stats["total"] > 0 else 0
        stats["push_rate"] = stats["push"]/stats["total"] if stats["total"] > 0 else 0

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
        "variance": var,
        "action_stats": action_stats,
        "true_count_bins": tc_bins
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

        for a, stats in action_stats.items():
            config.log_message(f"\nAction: {a}")
            config.log_message(f"  Times Taken: {stats['total']}")
            config.log_message(f"  Win Rate: {stats['win_rate']*100:.2f}%")
            config.log_message(f"  Loss Rate: {stats['loss_rate']*100:.2f}%")
            config.log_message(f"  Push Rate: {stats['push_rate']*100:.2f}%")

        config.log_message("\nTrue Count Bins:")
        for b, stats in sorted(tc_bins.items(), key=lambda x: (x[0].startswith('≤'), x[0].startswith('≥'), x[0])):
            config.log_message(f"TC Bin: {b} | Hands: {stats['total']} | Win: {stats['win_rate']*100:.2f}% | Lose: {stats['loss_rate']*100:.2f}% | Push: {stats['push_rate']*100:.2f}%")

    return summary

