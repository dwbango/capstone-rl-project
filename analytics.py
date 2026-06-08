# analytics.py

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server usage
import matplotlib.pyplot as plt
import statistics
import config
import math
import scipy.stats as stats
from itertools import combinations
from collections import Counter

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
        self.strategy_decisions = []

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
        
    def log_strategy_decision(
        self,
        player_total,
        dealer_upcard,
        is_soft,
        is_pair,
        hand_size,
        true_count,
        available_actions,
        chosen_action,
        shoe_number,
        hand_number
    ):
        """
        Logs each player decision with the game state at the time of action.
        Used to build observed strategy charts from actual simulation behavior.
        """
        self.strategy_decisions.append({
            "player_total": player_total,
            "dealer_upcard": dealer_upcard,
            "is_soft": is_soft,
            "is_pair": is_pair,
            "hand_size": hand_size,
            "true_count": true_count,
            "available_actions": list(available_actions),
            "chosen_action": chosen_action,
            "shoe_number": shoe_number,
            "hand_number": hand_number
        })

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


# ------------------------ Single-Run Plot Functions ------------------------

def plot_bankroll_over_time(bankroll_history):
    if not bankroll_history:
        return
    plt.figure(figsize=(10,5))
    plt.plot(range(len(bankroll_history)), bankroll_history, label='Bankroll')
    plt.title("Bankroll vs. Hand Number (Single Run)")
    plt.xlabel("Hand Number")
    plt.ylabel("Bankroll")
    plt.grid(True)
    plt.legend()
    plt.savefig('static/bankroll_history.png')
    plt.close()

def plot_ev_over_time(logger):
    records = logger.get_data()
    if not records:
        return
    cum_profit = 0.0
    x_vals = []
    ev_vals = []

    for i, r in enumerate(records):
        cum_profit += r["profit"]
        hand_index = i + 1
        avg_profit = cum_profit / hand_index
        x_vals.append(hand_index)
        ev_vals.append(avg_profit)

    plt.figure(figsize=(10,5))
    plt.plot(x_vals, ev_vals, label='EV (Profit/Hand)')
    plt.title("Average Profit/Hand vs. Hand Number (Single Run)")
    plt.xlabel("Hand Number")
    plt.ylabel("Average Profit/Hand")
    plt.grid(True)
    plt.legend()
    plt.savefig("static/ev_over_time.png")
    plt.close()

def plot_epsilon_convergence(logger):
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


# -------------------- Multi-Run / Compare-All Plot Functions --------------------

def plot_compare_bankroll(method_loggers):
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


# ---------------------- Summary/Stats Computation ----------------------

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

    final_bankroll = records[-1]["bankroll"] if records else config.STARTING_BANKROLL
    net_profit = final_bankroll - config.STARTING_BANKROLL
    wager = config.DEFAULT_WAGER
    ev_pct = (ev / wager) if wager else 0.0

    ev_rounded          = round(ev, 4)
    var_rounded         = round(var, 4)
    std_dev_rounded     = round(std_dev, 4)
    win_rate_rounded    = round(win_rate, 4)    
    loss_rate_rounded   = round(loss_rate, 4)
    push_rate_rounded   = round(push_rate, 4)
    ev_pct_rounded      = round(ev_pct, 4)
    final_bankroll_r    = round(final_bankroll, 2)  # 2 decimals for money
    net_profit_r        = round(net_profit, 2)      # money => 2 decimals
    
    summary = {
        "total_hands": total_hands,
        "wins": wins,
        "win_rate": win_rate_rounded,
        "losses": losses,
        "loss_rate": loss_rate_rounded,
        "pushes": pushes,
        "push_rate": push_rate_rounded,
        "final_bankroll": final_bankroll_r,
        "net_profit": net_profit_r,
        "EV_per_hand": ev_rounded,
        "EV_percent": ev_pct_rounded,
        "variance": var_rounded,
        "std_dev": std_dev_rounded
    }
    return summary


# ------------------ RL Strategy Charts (Hard/Soft/Pairs) ------------------

OBSERVED_ACTION_MAP = {
    None: -1,
    'hit': 0,
    'stand': 1,
    'double': 2,
    'split': 3
}

OBSERVED_ACTION_LABELS = ['N/A', 'hit', 'stand', 'double', 'split']

def _most_common_action_for_cell(strategy_decisions, player_total, dealer_upcard, is_soft, is_pair, hand_size=None):
    """
    Returns the most common observed action for a chart cell.
    Returns None if the state was not visited.
    """
    matching_actions = [
        d["chosen_action"]
        for d in strategy_decisions
        if d["player_total"] == player_total
        and d["dealer_upcard"] == dealer_upcard
        and d["is_soft"] == is_soft
        and d["is_pair"] == is_pair
        and (hand_size is None or d.get("hand_size") == hand_size)
    ]

    if not matching_actions:
        return None

    return Counter(matching_actions).most_common(1)[0][0]


def _plot_observed_strategy_chart(data, x_labels, y_labels, title, ylabel, filename):
    """
    Plots observed strategy data where -1 means unvisited.
    """
    cmap = plt.cm.get_cmap('viridis', 5)
    cmap.set_under('lightgray')

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, aspect='auto', vmin=-0.5, vmax=3.5)
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("Dealer Upcard")
    plt.ylabel(ylabel)
    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_ticks([-1, 0, 1, 2, 3])
    cbar.set_ticklabels(OBSERVED_ACTION_LABELS)

    plt.savefig(filename)
    plt.close()


def generate_all_strategy_charts_from_logger(logger):
    """
    Generate Hard/Soft/Pairs strategy charts from observed simulation decisions.
    Unvisited states are shown as N/A.
    """
    generate_hard_chart_from_logger(logger, 'static/strategy_chart_hard.png')
    generate_soft_chart_from_logger(logger, 'static/strategy_chart_soft.png')
    generate_pairs_chart_from_logger(logger, 'static/strategy_chart_pairs.png')


def generate_hard_chart_from_logger(logger, filename='static/strategy_chart_hard.png'):
    dealer_upcards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    player_totals = list(range(5, 21))
    data = []

    for p_total in player_totals:
        row = []
        for d_up in dealer_upcards:
            action = _most_common_action_for_cell(
                logger.strategy_decisions,
                player_total=p_total,
                dealer_upcard=d_up,
                is_soft=0,
                is_pair=False
            )
            row.append(OBSERVED_ACTION_MAP.get(action, -1))
        data.append(row)

    _plot_observed_strategy_chart(
        data,
        x_labels=dealer_upcards,
        y_labels=player_totals,
        title="Observed Strategy Chart - Hard Totals",
        ylabel="Player HARD Total",
        filename=filename
    )


def generate_soft_chart_from_logger(logger, filename='static/strategy_chart_soft.png'):
    dealer_upcards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    soft_totals = list(range(13, 21))
    data = []

    for p_total in soft_totals:
        row = []
        for d_up in dealer_upcards:
            action = _most_common_action_for_cell(
                logger.strategy_decisions,
                player_total=p_total,
                dealer_upcard=d_up,
                is_soft=1,
                is_pair=False
            )
            row.append(OBSERVED_ACTION_MAP.get(action, -1))
        data.append(row)

    _plot_observed_strategy_chart(
        data,
        x_labels=dealer_upcards,
        y_labels=soft_totals,
        title="Observed Strategy Chart - Soft Totals",
        ylabel="Player SOFT Total",
        filename=filename
    )


def generate_pairs_chart_from_logger(logger, filename='static/strategy_chart_pairs.png'):
    dealer_upcards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    pairs = [
        ('2,2', 4, 0),
        ('3,3', 6, 0),
        ('4,4', 8, 0),
        ('5,5', 10, 0),
        ('6,6', 12, 0),
        ('7,7', 14, 0),
        ('8,8', 16, 0),
        ('9,9', 18, 0),
        ('10,10', 20, 0),
        ('A,A', 12, 1),
    ]

    data = []
    labels = []

    for pair_label, p_total, is_soft in pairs:
        row = []
        for d_up in dealer_upcards:
            action = _most_common_action_for_cell(
                logger.strategy_decisions,
                player_total=p_total,
                dealer_upcard=d_up,
                is_soft=is_soft,
                is_pair=True,
                hand_size=2
            )
            row.append(OBSERVED_ACTION_MAP.get(action, -1))
        data.append(row)
        labels.append(pair_label)

    _plot_observed_strategy_chart(
        data,
        x_labels=dealer_upcards,
        y_labels=labels,
        title="Observed Strategy Chart - Pairs",
        ylabel="Pair",
        filename=filename
    )
    
    
def generate_all_strategy_charts(agent):
    if hasattr(agent, 'set_greedy'):
        agent.set_greedy()

    generate_hard_chart(agent, 'static/strategy_chart_hard.png')
    generate_soft_chart(agent, 'static/strategy_chart_soft.png')
    generate_pairs_chart(agent, 'static/strategy_chart_pairs.png')

ACTION_MAP = {'hit': 0, 'stand': 1}

def filter_chart_actions(p_total, is_soft, is_pair):
    valid_actions = ['hit','stand']
 #   if not is_pair:
  #      valid_actions.remove('split')
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

    # If any method has an empty list, skip ANOVA
    if any(len(lst) == 0 for lst in ev_lists):
        return {
            'anova_f': None,
            'anova_p': None,
            'pairwise': [],
            'error': "One or more methods has no data for ANOVA."
        }

    f_stat, p_val = stats.f_oneway(*ev_lists)

    pairwise_results = []
    alpha = 0.05
    if p_val < alpha and len(method_labels) > 2:
        combos = list(combinations(range(len(ev_lists)), 2))
        bonf_alpha = alpha / len(combos)

        for (i, j) in combos:
            groupA = ev_lists[i]
            groupB = ev_lists[j]
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


# ------------------- Confidence Intervals -------------------
def compute_confidence_interval(data, confidence=0.95):
    """
    Compute a confidence interval for the mean of the given data.
    Uses the t-distribution for smaller samples (assumes data ~ normal or large sample).
    
    Returns (lower_bound, upper_bound).
    If data is empty or there's an error, returns (None, None).
    """
    if len(data) < 2:
        return (None, None)

    mean_val = statistics.mean(data)
    std_err = stats.sem(data)  # standard error of the mean
    # For the t-distribution, degrees of freedom = len(data) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    margin = t_crit * std_err

    lower = mean_val - margin
    upper = mean_val + margin
    return (lower, upper)

def compute_confidence_intervals(method_ev_dict, confidence=0.95):
    """
    Given a dict of method -> list of EV values,
    computes the mean, std dev, and a confidence interval for each method.
    
    Returns a dict like:
      {
        'BasicStrategy': {
            'mean': 0.1234,
            'std_dev': 1.2345,
            'ci_lower': 0.12,
            'ci_upper': 0.45
        },
        ...
      }
    """
    result = {}
    for method, ev_list in method_ev_dict.items():
        if len(ev_list) == 0:
            result[method] = {
                "mean": None,
                "std_dev": None,
                "ci_lower": None,
                "ci_upper": None
            }
        else:
            mean_ev = statistics.mean(ev_list)
            std_dev = statistics.pstdev(ev_list)  # or stdev if population vs sample
            (ci_lo, ci_hi) = compute_confidence_interval(ev_list, confidence)
            result[method] = {
                "mean": mean_ev,
                "std_dev": std_dev,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi
            }
    return result