# strategy.py

import config
from environment import calculate_hand_value, is_pair, get_card_value, dealer_upcard_value, deal_card

def get_player_decision(hand, dealer_card, bankroll, current_wager, can_double, can_split_hand, splits_done, max_splits=config.MAX_SPLITS):
    player_value, is_soft = calculate_hand_value(hand)
    if player_value == 21:
        return 'stand'

    dealer_value = dealer_upcard_value(dealer_card)

    # Pair logic
    if is_pair(hand):
        rank = hand[0][0]
        pair_val = get_card_value(hand[0])
        if rank == 'A':
            if can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                if dealer_value in [5,6] and can_double and bankroll >= current_wager:
                    return 'double'
                return 'hit'

        if pair_val == 10:
            return 'stand'

        if pair_val == 9:
            if dealer_value in [2,3,4,5,6,8,9] and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'stand'

        if pair_val == 8:
            if can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                if dealer_value in [2,3,4,5,6]:
                    return 'stand'
                else:
                    return 'hit'

        if pair_val == 7:
            if dealer_value in [2,3,4,5,6,7] and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'hit'

        if pair_val == 6:
            if dealer_value in [2,3,4,5,6] and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'hit'

        if pair_val == 5:
            if dealer_value in range(2,10) and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'

        if pair_val == 4:
            if dealer_value in [5,6] and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'hit'

        if pair_val == 3:
            if dealer_value in range(2,8) and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'hit'

        if pair_val == 2:
            if dealer_value in range(2,8) and can_split_hand and splits_done < max_splits and bankroll >= current_wager:
                return 'split'
            else:
                return 'hit'

    # Soft totals
    if is_soft:
        if player_value == 20:
            return 'stand'
        if player_value == 19:
            if dealer_value == 6 and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'stand'
        if player_value == 18:
            if dealer_value in range(2,7) and can_double and bankroll >= current_wager:
                return 'double'
            elif dealer_value in [9,10,11]:
                return 'hit'
            else:
                return 'stand'
        if player_value == 17:
            if dealer_value in [3,4,5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 16:
            if dealer_value in [4,5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 15:
            if dealer_value in [4,5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 14:
            if dealer_value in [5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 13:
            if dealer_value in [5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'

    # Hard totals
    if not is_soft:
        if player_value >= 17:
            return 'stand'
        if player_value == 16:
            if dealer_value in [2,3,4,5,6]:
                return 'stand'
            else:
                return 'hit'
        if player_value == 15:
            if dealer_value in [2,3,4,5,6]:
                return 'stand'
            else:
                return 'hit'
        if player_value == 14:
            if dealer_value in [2,3,4,5,6]:
                return 'stand'
            else:
                return 'hit'
        if player_value == 13:
            if dealer_value in [2,3,4,5,6]:
                return 'stand'
            else:
                return 'hit'
        if player_value == 12:
            if dealer_value in [4,5,6]:
                return 'stand'
            else:
                return 'hit'
        if player_value == 11:
            if can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 10:
            if dealer_value in range(2,10) and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value == 9:
            if dealer_value in [3,4,5,6] and can_double and bankroll >= current_wager:
                return 'double'
            else:
                return 'hit'
        if player_value <= 8:
            return 'hit'

    return 'hit'

def player_action(deck, initial_hand, dealer_hand, bankroll, wager, running_count, max_splits=config.MAX_SPLITS):
    player_hands = [initial_hand]
    wagers = [wager]
    current_hand_index = 0
    splits_done = 0
    d_up = dealer_hand[0]

    while current_hand_index < len(player_hands):
        hand = player_hands[current_hand_index]

        # If hand has only one card, deal another
        if len(hand) == 1:
            card, running_count = deal_card(deck, running_count)
            hand.append(card)

        while True:
            hand_value, _ = calculate_hand_value(hand)
            if hand_value > 21:
                break

            can_double = (len(hand) == 2)
            can_split_hand = is_pair(hand)
            action = get_player_decision(hand, d_up, bankroll, wagers[current_hand_index], can_double, can_split_hand, splits_done, max_splits)

            if action == 'stand':
                break
            elif action == 'hit':
                card, running_count = deal_card(deck, running_count)
                hand.append(card)
            elif action == 'double':
                if can_double and bankroll >= wagers[current_hand_index]:
                    bankroll -= wagers[current_hand_index]
                    wagers[current_hand_index] *= 2
                    card, running_count = deal_card(deck, running_count)
                    hand.append(card)
                    break
                else:
                    # Couldn't double, just hit
                    card, running_count = deal_card(deck, running_count)
                    hand.append(card)
            elif action == 'split':
                if can_split_hand and splits_done < max_splits and bankroll >= wagers[current_hand_index]:
                    bankroll -= wagers[current_hand_index]
                    c1, c2 = hand[0], hand[1]
                    new_hand_1 = [c1]
                    new_hand_2 = [c2]
                    player_hands[current_hand_index] = new_hand_1
                    player_hands.insert(current_hand_index+1, new_hand_2)
                    wagers.append(wagers[current_hand_index])
                    splits_done += 1

                    card, running_count = deal_card(deck, running_count)
                    new_hand_1.append(card)
                    continue
                else:
                    card, running_count = deal_card(deck, running_count)
                    hand.append(card)

            hand_value, _ = calculate_hand_value(hand)
            if hand_value > 21:
                break
            if action in ['stand','double']:
                break

        current_hand_index += 1

    return player_hands, wagers, bankroll, running_count

def update_bankroll(bankroll, wager, outcome):
    if outcome == 'win':
        return bankroll + (wager * 2)
    elif outcome == 'push':
        return bankroll + wager
    elif outcome == 'lose':
        return bankroll
    else:
        raise ValueError("Invalid outcome provided.")

def initialize_bankroll(initial_amount=config.STARTING_BANKROLL):
    return initial_amount

def place_wager(bankroll, true_count_int=0):
    """
    If BETTING_STYLE == 'flat', always bet config.DEFAULT_WAGER.
    Otherwise, handle counts below -3, above 6, or in between -3..6 using conditionals.
    """
    if config.BETTING_STYLE == "flat":
        wager_amount = config.DEFAULT_WAGER
    else:
        # 'spread' approach with conditionals
        if true_count_int >= 7:
            wager_amount = config.BET_SPREAD_DICT.get(6, 10)
        elif true_count_int <= -4:
            wager_amount = config.BET_SPREAD_DICT.get(-3, 10)
        else:
            # Direct dictionary lookup if within range -3..6
            wager_amount = config.BET_SPREAD_DICT.get(true_count_int, 10)

    if wager_amount > bankroll:
        raise ValueError("Wager exceeds available bankroll.")

    return bankroll - wager_amount, wager_amount
