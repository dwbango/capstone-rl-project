# config.py
import sys

# Configuration Parameters
NUM_DECKS = 1
DEFAULT_WAGER = 10
STARTING_BANKROLL = 100000
SHUFFLE_POINT = 0.25
TOTAL_CARDS = 52 * NUM_DECKS
NUM_SHOES_TO_PLAY = 10000
MAX_SPLITS = 3
# DAS
# INSURANCE
# SURRENDER
# RESPLIT ACES
# RL METHOD

# Verbose output flag
verbose = True

# Count values for cards
COUNT_VALUES = {
    '2': +1, '3': +1, '4': +1, '5': +1, '6': +1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

def log_message(msg):
    """Logs a message if verbose mode is enabled."""
    if verbose:
        print(msg)
        
RL_METHOD = "QLearning"  #QLEARNING, SARSA