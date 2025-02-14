# main.py

import config
import environment
import strategy
import analytics
from rl_agent import QLearningAgent, SarsaAgent, BasicStrategyAgent, RandomAgent

def create_agent(actions):
    if config.RL_METHOD == "QLearning":
        return QLearningAgent(actions)
    elif config.RL_METHOD == "Sarsa":
        return SarsaAgent(actions)
    elif config.RL_METHOD == "BasicStrategy":
        return BasicStrategyAgent(actions)
    elif config.RL_METHOD == "Random":
        return RandomAgent(actions)
    else:
        raise ValueError(f"Unknown RL method: {config.RL_METHOD}")

def run_main_simulation(agent_override=None, num_shoes=None):
    """
    A refactoring of the original main() logic into a flexible function.

    :param agent_override: An already-created RL or other agent instance.
                          If None, we'll create a new one based on config.RL_METHOD.
    :param num_shoes:     If provided, temporarily override config.NUM_SHOES_TO_PLAY
                          for this simulation run.

    :return: (results_dict, agent, logger) just like your old main returned.
    """

    # Temporarily override NUM_SHOES_TO_PLAY if requested
    original_num_shoes = config.NUM_SHOES_TO_PLAY
    if num_shoes is not None:
        config.NUM_SHOES_TO_PLAY = num_shoes

    logger = analytics.DataLogger()
    bankroll = strategy.initialize_bankroll()

    # Allowed actions
    ACTIONS_RL = ['hit','stand']
    ACTIONS_BS = ['hit','stand','double','split']

    # Pick which agent to use
    if agent_override is not None:
        agent = agent_override
    else:
        if config.RL_METHOD in ["BasicStrategy", "Random"]:
            agent = create_agent(ACTIONS_BS)
        else:
            agent = create_agent(ACTIONS_RL)

    running_count = 0
    true_count_int = 0
    shoes_played = 0

    # We increment this *once* per initial deal (ignoring splits).
    hands_played = 0
    splits_done = 0

    deck = environment.shuffle_deck(environment.create_deck())
    dealt_cards_this_shoe = []

    # ------------------- START of your existing while loop logic -------------------
    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:
        # Count one new initial deal
        hands_played += 1

        shoe_number = shoes_played + 1
        hand_starting_bankroll = bankroll

        # Place a bet
        try:
            bankroll, wager = strategy.place_wager(bankroll, true_count_int)
        except ValueError:
            # Not enough bankroll or invalid bet => end
            break

        original_bet = wager
        start_of_hand_tc = true_count_int
        start_decks_remaining = len(deck) / 52.0

        # Deal initial hands
        player_hand, dealer_hand, running_count, true_count_int = environment.deal_initial_hands(deck, running_count)

        # Track the 4 initial dealt cards
        for c in (player_hand + dealer_hand):
            dealt_cards_this_shoe.append(c)

        # Single-string dealer upcard => "Value-Suit"
        raw_upcard = dealer_hand[0]
        dealer_upcard_str = f"{raw_upcard[0]}-{raw_upcard[1]}"

        # Player's first 2 cards
        p1 = player_hand[0]
        p2 = player_hand[1]
        player_card_1_str = f"{p1[0]}-{p1[1]}"
        player_card_2_str = f"{p2[0]}-{p2[1]}"

        # Check blackjacks
        player_blackjack = environment.is_blackjack(player_hand)
        dealer_blackjack = environment.is_blackjack(dealer_hand)

        # Evaluate player's initial 2-card softness
        player_value, is_soft = environment.calculate_hand_value(player_hand)
        initial_soft_hand = is_soft

        # Is it a pair?
        is_pair = environment.can_split(player_hand)

        # Evaluate dealer's initial value
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)

        player_hands = [player_hand]
        wagers = [wager]
        current_hand_index = 0

        did_split = False
        did_double = False

        # ----------- Scenario (1): Dealer has BJ -----------
        if dealer_blackjack:
            outcome = 'push' if player_blackjack else 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, outcome)
            final_profit = bankroll - hand_starting_bankroll

            dealer_final_val = environment.calculate_hand_value(dealer_hand)[0]
            player_final_val = environment.calculate_hand_value(player_hand)[0]

            did_player_bust = (player_final_val > 21)
            did_dealer_bust = (dealer_final_val > 21)
            final_player_hand_size = len(player_hand)
            final_dealer_hand_size = len(dealer_hand)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in player_hand)
            final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)

            logger.log_hand(
                outcome=outcome,
                profit=final_profit,
                bankroll=bankroll,
                actions_taken=[],
                starting_true_count=start_of_hand_tc,
                starting_decks_remaining=start_decks_remaining,
                dealer_actions=[],
                dealer_final_total=dealer_final_val,
                player_final_total=player_final_val,
                dealer_blackjack=dealer_blackjack,
                player_blackjack=player_blackjack,
                shoe_number=shoe_number,
                original_bet=original_bet,
                did_split=False,
                did_double=False,
                dealer_upcard=dealer_upcard_str,
                initial_soft_hand=initial_soft_hand,
                player_card_1=player_card_1_str,
                player_card_2=player_card_2_str,
                # Additional fields
                is_pair=is_pair,
                did_player_bust=did_player_bust,
                did_dealer_bust=did_dealer_bust,
                final_player_hand_size=final_player_hand_size,
                final_dealer_hand_size=final_dealer_hand_size,
                final_player_cards=final_player_cards,
                final_dealer_cards=final_dealer_cards
            )

            if hasattr(agent, 'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

        # ----------- Scenario (2): Dealer not BJ, but player is -----------
        if player_blackjack:
            winnings = wager * 1.5
            bankroll += (winnings + wager)
            outcome = 'win'
            final_profit = bankroll - hand_starting_bankroll

            dealer_final_val = environment.calculate_hand_value(dealer_hand)[0]
            player_final_val = environment.calculate_hand_value(player_hand)[0]

            did_player_bust = (player_final_val > 21)
            did_dealer_bust = (dealer_final_val > 21)
            final_player_hand_size = len(player_hand)
            final_dealer_hand_size = len(dealer_hand)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in player_hand)
            final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)

            logger.log_hand(
                outcome=outcome,
                profit=final_profit,
                bankroll=bankroll,
                actions_taken=[],
                starting_true_count=start_of_hand_tc,
                starting_decks_remaining=start_decks_remaining,
                dealer_actions=[],
                dealer_final_total=dealer_final_val,
                player_final_total=player_final_val,
                dealer_blackjack=dealer_blackjack,
                player_blackjack=player_blackjack,
                shoe_number=shoe_number,
                original_bet=original_bet,
                did_split=False,
                did_double=False,
                dealer_upcard=dealer_upcard_str,
                initial_soft_hand=initial_soft_hand,
                player_card_1=player_card_1_str,
                player_card_2=player_card_2_str,
                is_pair=is_pair,
                did_player_bust=did_player_bust,
                did_dealer_bust=did_dealer_bust,
                final_player_hand_size=final_player_hand_size,
                final_dealer_hand_size=final_dealer_hand_size,
                final_player_cards=final_player_cards,
                final_dealer_cards=final_dealer_cards
            )

            if hasattr(agent, 'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

        # ----------- Scenario (3): Normal flow (may split/double) -----------
        actions_this_hand = []
        dealer_actions = []
        last_state = None
        last_action = None

        while current_hand_index < len(player_hands):
            phand = player_hands[current_hand_index]
            while True:
                pval, is_soft = environment.calculate_hand_value(phand)
                if pval > 21:
                    break

                if config.RL_METHOD == "BasicStrategy":
                    available_actions = ['hit','stand']
                    can_double = (len(phand) == 2 and bankroll >= wagers[current_hand_index])
                    can_split_hand = (
                        environment.can_split(phand)
                        and bankroll >= wagers[current_hand_index]
                        and splits_done < config.MAX_SPLITS
                    )
                    if can_double:
                        available_actions.append('double')
                    if can_split_hand:
                        available_actions.append('split')
                elif config.RL_METHOD == "Random":
                    available_actions = ['hit','stand']
                    can_double = (len(phand) == 2 and bankroll >= wagers[current_hand_index])
                    can_split_hand = (
                        environment.can_split(phand)
                        and bankroll >= wagers[current_hand_index]
                        and splits_done < config.MAX_SPLITS
                    )
                    if can_double:
                        available_actions.append('double')
                    if can_split_hand:
                        available_actions.append('split')
                else:
                    # QLearning / Sarsa => only 'hit','stand'
                    available_actions = ['hit','stand']

                state = (pval, dealer_value, 1 if is_soft else 0, true_count_int)

                if config.RL_METHOD == "BasicStrategy":
                    action = agent.choose_action(
                        state, available_actions,
                        phand, dealer_hand,
                        bankroll, wagers[current_hand_index],
                        splits_done
                    )
                elif config.RL_METHOD == "Random":
                    action = agent.choose_action(state, available_actions)
                else:
                    # QLearning / Sarsa
                    if 'double' in available_actions:
                        available_actions.remove('double')
                    if 'split' in available_actions:
                        available_actions.remove('split')
                    action = agent.choose_action(state, available_actions)

                actions_this_hand.append(action)
                last_state = state
                last_action = action

                # Execute chosen action
                if action == 'hit':
                    card, running_count, true_count_int = environment.deal_card(deck, running_count)
                    phand.append(card)
                    dealt_cards_this_shoe.append(card)
                    pval, is_soft = environment.calculate_hand_value(phand)
                    if pval > 21:
                        break

                elif action == 'stand':
                    break

                elif action == 'double' and config.RL_METHOD in ["BasicStrategy","Random"]:
                    if len(phand) == 2 and bankroll >= wagers[current_hand_index]:
                        did_double = True
                        bankroll -= wagers[current_hand_index]
                        wagers[current_hand_index] *= 2
                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        phand.append(card)
                        dealt_cards_this_shoe.append(card)
                    break

                elif action == 'split' and config.RL_METHOD in ["BasicStrategy","Random"]:
                    if (
                        environment.can_split(phand)
                        and splits_done < config.MAX_SPLITS
                        and bankroll >= wagers[current_hand_index]
                    ):
                        did_split = True
                        bankroll -= wagers[current_hand_index]
                        c1, c2 = phand[0], phand[1]
                        new_hand_1 = [c1]
                        new_hand_2 = [c2]
                        player_hands[current_hand_index] = new_hand_1
                        player_hands.insert(current_hand_index+1, new_hand_2)
                        wagers.insert(current_hand_index+1, wagers[current_hand_index])
                        splits_done += 1

                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        new_hand_1.append(card)
                        dealt_cards_this_shoe.append(card)
                    else:
                        # invalid split => just hit once
                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        phand.append(card)
                        dealt_cards_this_shoe.append(card)
                        pval, is_soft = environment.calculate_hand_value(phand)
                        if pval > 21:
                            break

            current_hand_index += 1

        # Dealer must play if not all player hands bust
        dealer_must_play = any(environment.calculate_hand_value(h)[0] <= 21 for h in player_hands)
        if dealer_must_play:
            dealer_hand_copy = dealer_hand.copy()
            dealer_hand_copy, dealer_value, running_count, dealer_hits = \
                environment.dealer_turn(dealer_hand_copy, deck, running_count)
            if dealer_hits > 0:
                dealer_actions = ['hit'] * dealer_hits
                if dealer_value <= 21:
                    dealer_actions.append('stand')
            else:
                dealer_actions = ['stand']
        else:
            dealer_actions = ['stand']

        # Evaluate each splitted hand
        for i, phand in enumerate(player_hands):
            pval, _ = environment.calculate_hand_value(phand)
            w = wagers[i]

            if pval <= 21:
                if dealer_value > 21 or pval > dealer_value:
                    outcome = 'win'
                    bankroll = strategy.update_bankroll(bankroll, w, 'win')
                elif pval < dealer_value:
                    outcome = 'lose'
                    bankroll = strategy.update_bankroll(bankroll, w, 'lose')
                else:
                    outcome = 'push'
                    bankroll = strategy.update_bankroll(bankroll, w, 'push')
            else:
                outcome = 'lose'
                bankroll = strategy.update_bankroll(bankroll, w, 'lose')

            final_profit = bankroll - hand_starting_bankroll

            # RL updates if QLearning/Sarsa
            if config.RL_METHOD in ["QLearning","Sarsa"] and last_action is not None and last_state is not None:
                if outcome == 'win':
                    reward = 1.0
                elif outcome == 'push':
                    reward = 0.0
                else:
                    reward = -1.0

                next_state = (0,0,0,0)  # terminal
                if config.RL_METHOD == "QLearning":
                    agent.update(last_state, last_action, reward, next_state)
                else:  # Sarsa
                    old_q = agent.get_q_value(last_state, last_action)
                    new_q = old_q + agent.alpha * (reward - old_q)
                    agent.q_table[(last_state, last_action)] = new_q
                    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

            dealer_final_val = dealer_value
            player_final_val = environment.calculate_hand_value(phand)[0]

            did_player_bust = (player_final_val > 21)
            did_dealer_bust = (dealer_final_val > 21)
            final_player_hand_size = len(phand)

            if dealer_must_play:
                final_dealer_hand_size = len(dealer_hand_copy)
                final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand_copy)
            else:
                final_dealer_hand_size = len(dealer_hand)
                final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in phand)

            logger.log_hand(
                outcome=outcome,
                profit=final_profit,
                bankroll=bankroll,
                actions_taken=actions_this_hand,
                starting_true_count=start_of_hand_tc,
                starting_decks_remaining=start_decks_remaining,
                dealer_actions=dealer_actions,
                dealer_final_total=dealer_final_val,
                player_final_total=player_final_val,
                dealer_blackjack=dealer_blackjack,
                player_blackjack=player_blackjack,
                shoe_number=shoe_number,
                original_bet=original_bet,
                did_split=did_split,
                did_double=did_double,
                dealer_upcard=dealer_upcard_str,
                initial_soft_hand=initial_soft_hand,
                player_card_1=player_card_1_str,
                player_card_2=player_card_2_str,
                is_pair=is_pair,
                did_player_bust=did_player_bust,
                did_dealer_bust=did_dealer_bust,
                final_player_hand_size=final_player_hand_size,
                final_dealer_hand_size=final_dealer_hand_size,
                final_player_cards=final_player_cards,
                final_dealer_cards=final_dealer_cards
            )

        # Possibly reshuffle
        running_count, old_shoes_played = running_count, shoes_played
        running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
        if shoes_played > old_shoes_played:
            logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
            dealt_cards_this_shoe = []
    # ------------------- END of your existing while loop logic -------------------

    # End of simulation
    summary = analytics.print_summary(logger, total_deals=hands_played)

    bankroll_history = logger.get_bankroll_history()
    if bankroll_history:
        analytics.plot_bankroll_over_time(bankroll_history)

    if config.RL_METHOD in ["QLearning","Sarsa"]:
        analytics.plot_epsilon_convergence(logger)

    # Restore config.NUM_SHOES_TO_PLAY if we overrode it
    config.NUM_SHOES_TO_PLAY = original_num_shoes

    return {
        "hands_played": hands_played,
        "shoes_played": shoes_played,
        "summary": summary
    }, agent, logger


def main():
    """
    Old entry point, now just calls run_main_simulation()
    so that "python main.py" still does a single normal simulation
    with the current config in config.py.
    """
    return run_main_simulation(agent_override=None, num_shoes=None)