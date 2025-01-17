# analytics.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import statistics
import config
import io

class DataLogger:
    def __init__(self):
        self.records = []
        self.hand_counter = 0
        self.shoe_records = []
        self.epsilon_values = []  # To track epsilon over time

    def log_hand(self, outcome, profit, bankroll, actions_taken, starting_true_count, starting_decks_remaining, dealer_actions=[]):
        self.records.append({
            "hand_number": self.hand_counter,
            "outcome": outcome,
            "profit": profit,
            "bankroll": bankroll,
            "actions_taken": actions_taken,
            "dealer_actions": dealer_actions,
            "starting_true_count": starting_true_count,
            "starting_decks_remaining": starting_decks_remaining
        })
        self.hand_counter += 1

    def log_shoe_data(self, shoe_number, card_order):
        self.shoe_records.append({
            "shoe_number": shoe_number,
            "card_order": card_order
        })

    def log_epsilon(self, epsilon):
        # Call this method from main after each hand if agent has epsilon
        self.epsilon_values.append(epsilon)

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

def plot_epsilon_convergence(logger: 'DataLogger'):
    # Plot epsilon over time if we have epsilon values
    if not logger.epsilon_values:
        return
    plt.figure(figsize=(10,5))
    plt.plot(range(len(logger.epsilon_values)), logger.epsilon_values, label='Epsilon')
    plt.title("Epsilon Convergence Over Time")
    plt.xlabel("Hand Number")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    plt.savefig('static/epsilon_convergence.png')
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
        "risk_of_ruin": None,  # Placeholder: no calculation
        "action_stats": action_stats,
        "true_count_bins": tc_bins
    }

    return summary

def generate_strategy_chart_plot(agent):
    # Only 'hit' and 'stand' actions
    if hasattr(agent, 'set_greedy'):
        agent.set_greedy()

    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    player_totals = list(range(5, 22))
    # Only hit and stand for chart
    ACTIONS = ['hit', 'stand']
    true_count_int = 0
    is_soft = 0

    action_map = {'hit':0, 'stand':1}
    data = []
    for p_total in player_totals:
        row = []
        for d_up in dealer_upcards:
            state = (p_total, d_up, is_soft, true_count_int)
            action = agent.choose_action(state, ACTIONS)
            row.append(action_map[action])
        data.append(row)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(player_totals)), player_totals)
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player Total")
    plt.title("Strategy Chart (Hard Totals)")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1])
    cbar.set_ticklabels(['hit','stand'])
    plt.savefig('static/strategy_chart.png')
    plt.close()


