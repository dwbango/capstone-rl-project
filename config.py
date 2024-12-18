# config.py
import sys

# Configuration Parameters
NUM_DECKS = 2
DEFAULT_WAGER = 10
STARTING_BANKROLL = 1000000
SHUFFLE_POINT = 0.25
TOTAL_CARDS = 52 * NUM_DECKS
NUM_SHOES_TO_PLAY = 100
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
        
RL_METHOD = "BasicStrategy"  #QLEARNING, Sarsa, BasicStrategy