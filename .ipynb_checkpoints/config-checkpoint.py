# config.py
import sys

# Configuration Parameters
NUM_DECKS = 2
DEFAULT_WAGER = 10
STARTING_BANKROLL = 1000000
SHUFFLE_POINT = 0.25
TOTAL_CARDS = 52 * NUM_DECKS
NUM_SHOES_TO_PLAY = 10 #750,000
MAX_SPLITS = 3
# DAS
# INSURANCE
# SURRENDER
# RESPLIT ACES

# Verbose output flag
verbose = False

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

# RL method can be: QLearning, Sarsa, or BasicStrategy
RL_METHOD = "BasicStrategy"

# ---------------------------
# Betting Variables
# ---------------------------
# Toggle between "flat" or "spread" betting.
BETTING_STYLE = "flat"

# Default dictionary for a spread from TC -3 up to +6.
# Each key = integer true count; each value = bet amount.
BET_SPREAD_DICT = {
    -3: 10,
    -2: 10,
    -1: 10,
     0: 10,
     1: 10,
     2: 25,
     3: 50,
     4: 75,
     5: 100,
     6: 125
}
