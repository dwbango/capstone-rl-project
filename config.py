# config.py

NUM_DECKS = 2
DEFAULT_WAGER = 25
STARTING_BANKROLL = 10000
SHUFFLE_POINT = 0.25
TOTAL_CARDS = 52 * NUM_DECKS
NUM_SHOES_TO_PLAY = 1000

COUNT_VALUES = {
    '2': +1, '3': +1, '4': +1, '5': +1, '6': +1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

verbose = False

