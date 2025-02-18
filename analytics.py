# analytics.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server usage
import matplotlib.pyplot as plt
import statistics
import config
import math
import scipy.stats as stats
from itertools import combinations

class DataLogger:
    """
    Logs hand-level data (profit, bankroll, actions, etc.) 
    plus shoe records and optional epsilon values for RL.
    """
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
        shoe_number=None,
        original_bet=None,
        did_split=False,
        did_double=False,
        dealer_upcard=None,
        initial_soft_hand=None,
        player_card_1=None,
        player_card_2=None,
        is_pair=None,
        did_player_bust=None,
        did_dealer_bust=None,
        final_player_hand_size=None,
        final_dealer_hand_size=None,
        final_player_cards=None,
        final_dealer_cards=None
    ):
        """
        Records details about each final outcome (including splitted hands).
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
            "shoe_number": shoe_number,
            "original_bet": original_bet,
            "did_split": did_split,
            "did_double": did_double,
            "dealer_upcard": dealer_upcard,
            "initial_soft_hand": initial_soft_hand,
            "player_card_1": player_card_1,
            "player_card_2": player_card_2,
            "is_pair": is_pair,
            "did_player_bust": did_player_bust,
            "did_dealer_bust": did_dealer_bust,
            "final_player_hand_size": final_player_hand_size,
            "final_dealer_hand_size": final_dealer_hand_size,
            "final_player_cards": final_player_cards,
            "final_dealer_cards": final_dealer_cards
        })
        self.hand_counter += 1

    def log_shoe_data(self, shoe_number, card_order):
        """
        Store data about each shoe (like the entire card order).
        """
        self.shoe_records.append({
            "shoe_number": shoe_number,
            "card_order": card_order
        })

    def log_epsilon(self, epsilon):
        """
        For RL methods: track epsilon after each update or hand.
        """
        self.epsilon_values.append(epsilon)

    def get_data(self):
        return list(self.records)

    def get_outcomes(self):
        return [r["outcome"] for r in self.records]

    def get_profits(self):
        return [r["profit"] for r in self.records]

    def get_bankroll_history(self):
        """
        Returns a list of bankroll values in the order they were recorded (hand by hand).
        """
        return [r["bankroll"] for r in self.records]

    def get_counts(self):
        """
        Returns (wins, losses, pushes) across all final outcomes in the records.
        """
        wins = sum(1 for r in self.records if r["outcome"] == "win")
        losses = sum(1 for r in self.records if r["outcome"] == "lose")
        pushes = sum(1 for r in self.records if r["outcome"] == "push")
        return wins, losses, pushes

# --------------- Single-Run Plot Functions ---------------

def plot_bankroll_over_time(bankroll_history):
    """
    Plots a single line for bankroll over time (one data logger).
    Saves as 'static/bankroll_history.png'.
    """
    if not bankroll_history:
        return
    plt.figure(figsize=(10,5))
    plt.plot(range(len(bankroll_history)), bankroll_history, label='Bankroll')
    plt.title("Bankroll Over Time (Single Run)")
    plt.xlabel("Hand Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.legend()
    plt.savefig('static/bankroll_history.png')
    plt.close()

def plot_ev_over_time(logger):
    """
    Plots the running average EV over time for a single DataLogger.
    Saves as 'static/ev_over_time.png'.
    """
    records = logger.get_data()
    if not records:
        return
    cum_profit = 0.0
    x_vals = []
    ev_vals = []

    # Each record has a "profit"
    for i, r in enumerate(records):
        cum_profit += r["profit"]
        hand_index = i + 1
        avg_profit = cum_profit / hand_index  # average profit per hand so far
        x_vals.append(hand_index)
        ev_vals.append(avg_profit)

    plt.figure(figsize=(10,5))
    plt.plot(x_vals, ev_vals, label='EV (Profit/Hand)')
    plt.title("EV Over Time (Single Run)")
    plt.xlabel("Hand Number")
    plt.ylabel("Average Profit/Hand")
    plt.grid(True)
    plt.legend()
    plt.savefig("static/ev_over_time.png")
    plt.close()

def plot_epsilon_convergence(logger):
    """
    If using QLearning or Sarsa, plot epsilon vs. hand number.
    Saves as 'static/epsilon_convergence.png'.
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

# --------------- Multi-Run / Compare-All Plot Functions ---------------

def plot_compare_bankroll(method_loggers):
    """
    Expecting a dict: { 'BasicStrategy': DataLogger, 'Random': DataLogger, ... }
    We'll plot each DataLogger's bankroll history in a single figure.
    
    Saves as 'static/compare_bankroll.png'.
    """
    if not method_loggers:
        return

    plt.figure(figsize=(10,5))

    for method, logger in method_loggers.items():
        if not logger or not hasattr(logger, 'get_bankroll_history'):
            continue
        bankroll_history = logger.get_bankroll_history()
        if not bankroll_history:
            continue

        x_vals = range(len(bankroll_history))
        plt.plot(x_vals, bankroll_history, label=method)

    plt.title("Bankroll Over Time - Compare All Methods")
    plt.xlabel("Hand Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.legend()
    plt.savefig('static/compare_bankroll.png')
    plt.close()

def plot_compare_ev(method_loggers):
    """
    Expecting a dict: { 'BasicStrategy': DataLogger, 'Random': DataLogger, ... }
    We'll compute a running-average profit per hand for each logger and plot them.
    
    Saves as 'static/ev_compare.png'.
    """
    if not method_loggers:
        return

    plt.figure(figsize=(10,5))

    for method, logger in method_loggers.items():
        if not logger or not hasattr(logger, 'get_data'):
            continue
        records = logger.get_data()
        if not records:
            continue

        cum_profit = 0.0
        ev_vals = []
        for i, r in enumerate(records):
            cum_profit += r["profit"]
            ev_vals.append(cum_profit / (i+1))  # average profit/hand so far

        x_vals = range(len(ev_vals))
        plt.plot(x_vals, ev_vals, label=method)

    plt.title("EV Over Time - Compare All Methods")
    plt.xlabel("Hand Number")
    plt.ylabel("Average Profit/Hand")
    plt.grid(True)
    plt.legend()
    plt.savefig("static/ev_compare.png")
    plt.close()

# --------------- Summary/Stats Computation ---------------

def compute_ev_per_hand(profits):
    if not profits:
        return 0.0
    return sum(profits) / len(profits)

def compute_outcome_rates(wins, losses, pushes, total_hands):
    if total_hands == 0:
        return 0.0, 0.0, 0.0
    return wins / total_hands, losses / total_hands, pushes / total_hands

def compute_variance(profits):
    if len(profits) <= 1:
        return 0.0
    return statistics.pvariance(profits)

def print_summary(logger, total_deals=None):
    """
    Creates a summary dict.
    If total_deals is provided, we use it for the denominators, otherwise len(records).
    """
    records = logger.get_data()

    if total_deals is not None:
        total_hands = total_deals
    else:
        total_hands = len(records)

    wins, losses, pushes = logger.get_counts()
    profits = [r["profit"] for r in records]

    ev = compute_ev_per_hand(profits)
    var = compute_variance(profits)
    std_dev = math.sqrt(var) if var > 0 else 0.0

    win_rate, loss_rate, push_rate = compute_outcome_rates(wins, losses, pushes, total_hands)

    # final bankroll is from last record or fallback to config.STARTING_BANKROLL
    final_bankroll = records[-1]["bankroll"] if records else config.STARTING_BANKROLL
    net_profit = final_bankroll - config.STARTING_BANKROLL
    wager = config.DEFAULT_WAGER
    ev_pct = (ev / wager) if wager else 0.0

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
        "std_dev": std_dev
    }
    return summary

# --------------- RL Strategy Charts (Hard/Soft/Pairs) ---------------

def generate_all_strategy_charts(agent):
    """
    For QLearning/Sarsa agents: create 3 strategy charts from Q-values.
    Saves: 'strategy_chart_hard.png', 'strategy_chart_soft.png', 'strategy_chart_pairs.png'.
    """
    if hasattr(agent, 'set_greedy'):
        agent.set_greedy()

    generate_hard_chart(agent, 'static/strategy_chart_hard.png')
    generate_soft_chart(agent, 'static/strategy_chart_soft.png')
    generate_pairs_chart(agent, 'static/strategy_chart_pairs.png')

ACTION_MAP = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3}

def filter_chart_actions(p_total, is_soft, is_pair):
    valid_actions = ['hit','stand','double','split']
    if not is_pair:
        valid_actions.remove('split')
    return valid_actions

def generate_hard_chart(agent, filename='static/strategy_chart_hard.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    player_totals = range(5,21)
    data = []

    for p_total in player_totals:
        row = []
        is_soft = 0
        is_pair = False
        for d_up in dealer_upcards:
            actions = filter_chart_actions(p_total, is_soft, is_pair)
            state = (p_total, d_up, is_soft, 0)
            action = agent.choose_action(state, actions)
            row.append(ACTION_MAP.get(action, 0))
        data.append(row)

    plt.figure(figsize=(8,6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(player_totals)), list(player_totals))
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player HARD Total")
    plt.title("Strategy Chart - Hard Totals")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

def generate_soft_chart(agent, filename='static/strategy_chart_soft.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
    soft_totals = range(13,21)
    data = []

    for p_total in soft_totals:
        row = []
        is_soft = 1
        is_pair = False
        for d_up in dealer_upcards:
            actions = filter_chart_actions(p_total, is_soft, is_pair)
            state = (p_total, d_up, is_soft, 0)
            action = agent.choose_action(state, actions)
            row.append(ACTION_MAP.get(action, 0))
        data.append(row)

    plt.figure(figsize=(8,6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(soft_totals)), list(soft_totals))
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player SOFT Total")
    plt.title("Strategy Chart - Soft Totals")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

def generate_pairs_chart(agent, filename='static/strategy_chart_pairs.png'):
    dealer_upcards = [2,3,4,5,6,7,8,9,10,11]
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
            row.append(ACTION_MAP.get(action, 0))
        data.append(row)
        labels.append(pair_str)

    plt.figure(figsize=(8,6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.xticks(range(len(dealer_upcards)), dealer_upcards)
    plt.yticks(range(len(pairs)), labels)
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Pair")
    plt.title("Strategy Chart - Pairs")
    cbar = plt.colorbar()
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['hit','stand','double','split'])
    plt.savefig(filename)
    plt.close()

# --------------- ANOVA + Post Hoc Testing ---------------

def run_anova_and_posthoc(method_ev_dict):
    """
    method_ev_dict: dict of { 'BasicStrategy': [EV1, EV2, ...], 'Random': [...], ... }
    
    1) Performs a one-way ANOVA across all methods' EV lists.
    2) If p < 0.05 and more than 2 methods, does pairwise t-tests (Bonferroni).

    Returns a dict with:
       'anova_f'   -> F-statistic
       'anova_p'   -> p-value
       'pairwise'  -> list of pairwise test results, each containing:
                      { 'method1', 'method2', 't_stat', 'p_val', 'significant_after_bonf': bool }
    """
    method_labels = list(method_ev_dict.keys())
    ev_lists = list(method_ev_dict.values())

    # Basic check: if any list is empty, we skip stats
    if any(len(lst) == 0 for lst in ev_lists):
        return {
            'anova_f': None,
            'anova_p': None,
            'pairwise': [],
            'error': "One or more methods has no data for ANOVA."
        }

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*ev_lists)

    # Post-hoc results
    pairwise_results = []
    alpha = 0.05

    # If significant and at least 2 combos:
    if p_val < alpha and len(method_labels) > 2:
        combos = list(combinations(range(len(ev_lists)), 2))
        bonf_alpha = alpha / len(combos)

        for (i, j) in combos:
            groupA = ev_lists[i]
            groupB = ev_lists[j]
            # Two-sample t-test (Welch's)
            t_stat, p_ttest = stats.ttest_ind(groupA, groupB, equal_var=False)
            sig = p_ttest < bonf_alpha
            pairwise_results.append({
                'method1': method_labels[i],
                'method2': method_labels[j],
                't_stat': t_stat,
                'p_val': p_ttest,
                'significant_after_bonf': sig
            })

    return {
        'anova_f': f_stat,
        'anova_p': p_val,
        'pairwise': pairwise_results
    }