# analytics.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import statistics
import config
import io

class DataLogger:
    def __init__(self):
        self.records = []
        self.hand_counter = 0
        self.shoe_records = []
        self.epsilon_values = []

    def log_hand(
        self,
        outcome,
        profit,
        bankroll,
        actions_taken,
        starting_true_count,
        starting_decks_remaining,
        dealer_actions=[],
        dealer_final_total=None,
        player_final_total=None,
        dealer_blackjack=False,
        player_blackjack=False,
        # ------------- NEW FIELDS ----------------
        shoe_number=None,       # which shoe is this hand in?
        original_bet=None,      # the bet amount before doubling
        did_split=False,        # did the player split at least once?
        did_double=False        # did the player double at least once?
        # -----------------------------------------
    ):
        """
        Records details about a single final outcome (which could be
        a regular hand or a split outcome).
        """
        self.records.append({
            "hand_number": self.hand_counter,
            "outcome": outcome,
            "profit": profit,
            "bankroll": bankroll,
            "actions_taken": actions_taken,
            "dealer_actions": dealer_actions,
            "starting_true_count": starting_true_count,
            "starting_decks_remaining": starting_decks_remaining,
            "dealer_final_total": dealer_final_total,
            "player_final_total": player_final_total,
            "dealer_blackjack": dealer_blackjack,
            "player_blackjack": player_blackjack,

            # Extra fields
            "shoe_number": shoe_number,
            "original_bet": original_bet,
            "did_split": did_split,
            "did_double": did_double
        })
        self.hand_counter += 1

    def log_shoe_data(self, shoe_number, card_order):
        """
        Store data about each shoe (like the order of cards).
        """
        self.shoe_records.append({
            "shoe_number": shoe_number,
            "card_order": card_order
        })

    def log_epsilon(self, epsilon):
        self.epsilon_values.append(epsilon)

    def get_data(self):
        return self.records[:]

    def get_outcomes(self):
        return [r["outcome"] for r in self.records]

    def get_profits(self):
        return [r["profit"] for r in self.records]

    def get_bankroll_history(self):
        """
        Returns a list of bankroll values in the order they were recorded.
        Typically this is one entry per final hand (including splits).
        """
        return [r["bankroll"] for r in self.records]

    def get_counts(self):
        """
        Returns (wins, losses, pushes) across all final outcomes.
        """
        wins = sum(1 for r in self.records if r["outcome"] == "win")
        losses = sum(1 for r in self.records if r["outcome"] == "lose")
        pushes = sum(1 for r in self.records if r["outcome"] == "push")
        return wins, losses, pushes

# --------------------- Plotting & Summaries -----------------------

def plot_bankroll_over_time(bankroll_history):
    """
    Plots the bankroll over time (x-axis = each final hand outcome).
    If you want one point per initial deal, you'll need to log it differently.
    """
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
    """
    If using QLearning or Sarsa, plots epsilon vs. number of hands (updates).
    """
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
    """
    Expected value per final hand outcome.
    """
    if not profits:
        return 0.0
    return sum(profits) / len(profits)

def compute_outcome_rates(wins, losses, pushes, total_hands):
    """
    Win/loss/push rates. total_hands can be the number of initial deals
    (if ignoring splits), or the number of final outcomes, etc.
    """
    if total_hands == 0:
        return 0.0, 0.0, 0.0
    return wins / total_hands, losses / total_hands, pushes / total_hands

def compute_variance(profits):
    if len(profits) <= 1:
        return 0.0
    return statistics.pvariance(profits)

def tc_to_bin(tc):
    """
    Converts a true count integer to a string bin label for summary stats.
    """
    if tc <= -7:
        return "≤ -7"
    elif tc >= 7:
        return "≥ 7"
    else:
        return str(tc)

def print_summary(logger: 'DataLogger', total_deals=None):
    """
    Creates a summary dict of the simulation results.
    If total_deals is specified (the number of initial deals ignoring splits),
    we use that for 'total_hands' instead of len(logger.records).
    """
    records = logger.get_data()

    # total_hands can be forced to match the # of initial deals
    # if total_deals is provided; otherwise it falls back to the
    # number of final outcomes in logger.records.
    if total_deals is not None:
        total_hands = total_deals
    else:
        total_hands = len(records)

    wins, losses, pushes = logger.get_counts()
    profits = [r["profit"] for r in records]

    ev = compute_ev_per_hand(profits)
    var = compute_variance(profits)
    win_rate, loss_rate, push_rate = compute_outcome_rates(wins, losses, pushes, total_hands)

    final_bankroll = records[-1]["bankroll"] if records else config.STARTING_BANKROLL
    net_profit = final_bankroll - config.STARTING_BANKROLL
    wager = config.DEFAULT_WAGER
    ev_pct = (ev / wager) if wager else 0.0

    # Action-level stats
    action_stats = {}
    for r in records:
        outcome = r["outcome"]
        for a in r.get("actions_taken", []):
            if a not in action_stats:
                action_stats[a] = {"win":0, "lose":0, "push":0, "total":0}
            action_stats[a][outcome] += 1
            action_stats[a]["total"] += 1

    for a, stats in action_stats.items():
        if stats["total"] > 0:
            stats["win_rate"] = stats["win"] / stats["total"]
            stats["loss_rate"] = stats["lose"] / stats["total"]
            stats["push_rate"] = stats["push"] / stats["total"]
        else:
            stats["win_rate"] = 0
            stats["loss_rate"] = 0
            stats["push_rate"] = 0

    # True-count bins
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
        if stats["total"] > 0:
            stats["win_rate"] = stats["win"] / stats["total"]
            stats["loss_rate"] = stats["lose"] / stats["total"]
            stats["push_rate"] = stats["push"] / stats["total"]
        else:
            stats["win_rate"] = 0
            stats["loss_rate"] = 0
            stats["push_rate"] = 0

    return {
        "total_hands": total_hands,  # <= either user-supplied or len(records)
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
        "risk_of_ruin": None,   # Not currently implemented
        "action_stats": action_stats,
        "true_count_bins": tc_bins
    }

# ------------------ Chart Generation (unchanged) ------------------
ACTION_MAP = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3}

def filter_chart_actions(p_total, is_soft, is_pair):
    valid_actions = ['hit','stand','double','split']
    if not is_pair:
        valid_actions.remove('split')
    return valid_actions

def generate_hard_chart(agent, filename='static/strategy_chart_hard.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    player_totals = range(5,22)
    data = []
    for p_total in player_totals:
        is_soft = 0
        is_pair = False
        row = []
        for d_up in dealer_upcards:
            actions = filter_chart_actions(p_total, is_soft, is_pair)
            state = (p_total, d_up, is_soft, 0)
            action = agent.choose_action(state, actions)
            data_val = ACTION_MAP.get(action, 0)
            row.append(data_val)
        data.append(row)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(player_totals)), list(player_totals))
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player HARD Total (2-card)")
    plt.title("Strategy Chart - Hard Totals (2-card)")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

def generate_soft_chart(agent, filename='static/strategy_chart_soft.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    soft_totals = range(13,22)
    data = []
    for p_total in soft_totals:
        is_soft = 1
        is_pair = False
        row = []
        for d_up in dealer_upcards:
            actions = filter_chart_actions(p_total, is_soft, is_pair)
            state = (p_total, d_up, is_soft, 0)
            action = agent.choose_action(state, actions)
            data_val = ACTION_MAP.get(action, 0)
            row.append(data_val)
        data.append(row)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(soft_totals)), list(soft_totals))
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player SOFT Total (2-card)")
    plt.title("Strategy Chart - Soft Totals (2-card)")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

def generate_pairs_chart(agent, filename='static/strategy_chart_pairs.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    # Each tuple: (display_label, player_total, is_soft)
    pairs = [
        ('2,2',4,0),
        ('3,3',6,0),
        ('4,4',8,0),
        ('5,5',10,0),
        ('6,6',12,0),
        ('7,7',14,0),
        ('8,8',16,0),
        ('9,9',18,0),
        ('10,10',20,0),
        ('A,A',12,0),
    ]
    data = []
    labels = []
    for (pair_str, p_total, is_soft) in pairs:
        is_pair = True
        row = []
        for d_up in dealer_upcards:
            actions = filter_chart_actions(p_total, is_soft, is_pair)
            state = (p_total, d_up, is_soft, 0)
            action = agent.choose_action(state, actions)
            data_val = ACTION_MAP.get(action, 0)
            row.append(data_val)
        data.append(row)
        labels.append(pair_str)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(pairs)), labels)
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Pair (2-card)")
    plt.title("Strategy Chart - Pairs (2-card)")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

def generate_all_strategy_charts(agent):
    """
    Generates all 3 strategy charts if the agent is QLearning or Sarsa-based.
    For a BasicStrategy or Random agent, you wouldn't typically generate these charts.
    """
    if hasattr(agent, 'set_greedy'):
        agent.set_greedy()
    generate_hard_chart(agent, 'static/strategy_chart_hard.png')
    generate_soft_chart(agent, 'static/strategy_chart_soft.png')
    generate_pairs_chart(agent, 'static/strategy_chart_pairs.png')