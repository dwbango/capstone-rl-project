# environment.py
import random
import config

def create_deck():
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    single_deck = [(value, suit) for suit in suits for value in values]
    return single_deck * config.NUM_DECKS

def shuffle_deck(deck):
    random.shuffle(deck)
    return deck

def deal_card(deck, running_count):
    if len(deck) <= int(config.TOTAL_CARDS * config.SHUFFLE_POINT):
        config.log_message("Cut card reached! Reshuffle after this round.")

    card = deck.pop()
    card_val = config.COUNT_VALUES.get(card[0], 0)
    running_count += card_val

    # Compute true count
    decks_remaining = len(deck)/52.0
    if decks_remaining > 0:
        true_count = running_count/decks_remaining
    else:
        true_count = running_count
    true_count_int = int(round(true_count))

    return card, running_count, true_count_int

def deal_initial_hands(deck, running_count):
    # Deal to player: first card
    card, running_count, true_count_int = deal_card(deck, running_count)
    player_hand = [card]
    # Second card to player
    card, running_count, true_count_int = deal_card(deck, running_count)
    player_hand.append(card)
    # First card to dealer
    card, running_count, true_count_int = deal_card(deck, running_count)
    dealer_hand = [card]
    # Second card to dealer
    card, running_count, true_count_int = deal_card(deck, running_count)
    dealer_hand.append(card)

    # Return the hands, updated running_count, and the latest true_count_int
    return player_hand, dealer_hand, running_count, true_count_int

def reshuffle_if_needed(deck, running_count, shoes_played):
    if len(deck) <= int(config.TOTAL_CARDS * config.SHUFFLE_POINT):
        config.log_message("Reshuffling deck now...")
        deck[:] = shuffle_deck(create_deck())
        running_count = 0
        shoes_played += 1
        config.log_message(f"*** Completed Shoe {shoes_played} ***")
    return running_count, shoes_played

def calculate_hand_value(hand):
    value = 0
    aces_used_as_11 = 0
    card_values = {
        '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
        '8':8,'9':9,'10':10,'J':10,'Q':10,'K':10
    }

    for card, suit in hand:
        if card == 'A':
            value += 11
            aces_used_as_11 += 1
        else:
            value += card_values[card]

    while value > 21 and aces_used_as_11 > 0:
        value -= 10
        aces_used_as_11 -= 1

    is_soft = aces_used_as_11 > 0
    return value, is_soft

def is_blackjack(hand):
    value, _ = calculate_hand_value(hand)
    return len(hand) == 2 and value == 21

def can_split(hand):
    if len(hand) == 2:
        card_values = {
            '2': 2, '3': 3, '4':4, '5':5,'6':6,'7':7,
            '8':8,'9':9,'10':10,'J':10,'Q':10,'K':10,'A':11
        }
        return card_values[hand[0][0]] == card_values[hand[1][0]]
    return False

def dealer_turn(dealer_hand, deck, running_count):
    while True:
        dealer_value, is_soft = calculate_hand_value(dealer_hand)
        config.log_message(f"Dealer's Hand: {dealer_hand} | Value: {dealer_value}")

        if dealer_value > 17:
            break
        elif dealer_value == 17 and is_soft:
            card, running_count, tc = deal_card(deck, running_count)
            dealer_hand.append(card)
        elif dealer_value == 17:
            break
        else:
            card, running_count, tc = deal_card(deck, running_count)
            dealer_hand.append(card)

    dealer_value, _ = calculate_hand_value(dealer_hand)
    return dealer_hand, dealer_value, running_count

def is_pair(hand):
    return len(hand) == 2 and hand[0][0] == hand[1][0]

def get_card_value(card):
    card_rank = card[0]
    if card_rank in ['J','Q','K','10']:
        return 10
    elif card_rank == 'A':
        return 11
    else:
        return int(card_rank)

def dealer_upcard_value(card):
    if card[0] in ['J','Q','K','10']:
        return 10
    elif card[0] == 'A':
        return 11
    else:
        return int(card[0])

def determine_winner(player_value, dealer_value):
    config.log_message("\n--- Final Results ---")
    if player_value > 21:
        config.log_message("You busted. Dealer wins.")
    elif dealer_value > 21:
        config.log_message("Dealer busted. You win!")
    elif player_value > dealer_value:
        config.log_message("You win!")
    elif player_value < dealer_value:
        config.log_message("Dealer wins.")
    else:
        config.log_message("It's a push!")

