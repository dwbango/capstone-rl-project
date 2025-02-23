import config
import environment
import strategy
import analytics
from rl_agent import QLearningAgent, SarsaAgent, BasicStrategyAgent, RandomAgent

def create_agent(actions):
    """
    Factory function to build the appropriate agent
    based on config.RL_METHOD.
    """
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
    Runs one simulation (with the chosen RL method or a provided agent).

    :param agent_override: If not None, use this pre-created RL agent instead
                           of building a new one from config.
    :param num_shoes:      Temporarily override config.NUM_SHOES_TO_PLAY if not None.
    :return: (results_dict, agent, logger)
    """

    # Temporarily override the user-specified # of shoes
    original_num_shoes = config.NUM_SHOES_TO_PLAY
    if num_shoes is not None:
        config.NUM_SHOES_TO_PLAY = num_shoes

    # Create a fresh DataLogger for this run
    logger = analytics.DataLogger()

    # Initialize bankroll from config
    bankroll = strategy.initialize_bankroll()

    # Possible action sets
    ACTIONS_RL = ['hit', 'stand']         # For QLearning / Sarsa (only 2 actions)
    ACTIONS_BS = ['hit', 'stand', 'double', 'split']  # For BasicStrategy / Random

    # If an external agent is provided, use it. Otherwise, create based on config
    if agent_override is not None:
        agent = agent_override
    else:
        if config.RL_METHOD in ["BasicStrategy", "Random"]:
            agent = create_agent(ACTIONS_BS)
        else:
            agent = create_agent(ACTIONS_RL)

    # Prepare shoe + counters
    deck = environment.shuffle_deck(environment.create_deck())
    running_count = 0
    true_count_int = 0
    shoes_played = 0
    hands_played = 0
    splits_done = 0
    dealt_cards_this_shoe = []

    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:

        hands_played += 1
        shoe_number = shoes_played + 1
        hand_starting_bankroll = bankroll

        # Place a bet
        try:
            bankroll, wager = strategy.place_wager(bankroll, true_count_int)
        except ValueError:
            # Means bankroll < needed wager => effectively busted
            break

        original_bet = wager
        start_of_hand_tc = true_count_int
        start_decks_remaining = len(deck) / 52.0

        # Deal initial cards
        player_hand, dealer_hand, running_count, true_count_int = environment.deal_initial_hands(
            deck, running_count
        )
        for c in (player_hand + dealer_hand):
            dealt_cards_this_shoe.append(c)

        # Convert cards to strings for logging
        dealer_upcard_str = f"{dealer_hand[0][0]}-{dealer_hand[0][1]}"
        player_card_1_str = f"{player_hand[0][0]}-{player_hand[0][1]}"
        player_card_2_str = f"{player_hand[1][0]}-{player_hand[1][1]}"

        player_blackjack = environment.is_blackjack(player_hand)
        dealer_blackjack = environment.is_blackjack(dealer_hand)

        # Evaluate softness/pair
        player_value, is_soft = environment.calculate_hand_value(player_hand)
        initial_soft_hand = is_soft
        is_pair = environment.can_split(player_hand)
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)

        player_hands = [player_hand]
        wagers       = [wager]
        current_hand_index = 0

        did_split  = False
        did_double = False

        # ---------- Immediate Blackjacks ----------
        if dealer_blackjack:
            # Dealer has BJ => either push if player also has BJ, else lose
            outcome = 'push' if player_blackjack else 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, outcome)
            final_profit = bankroll - hand_starting_bankroll

            dealer_final_val = environment.calculate_hand_value(dealer_hand)[0]
            player_final_val = environment.calculate_hand_value(player_hand)[0]
            did_player_bust = (player_final_val > 21)
            did_dealer_bust = (dealer_final_val > 21)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in player_hand)
            final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(
                deck, running_count, shoes_played
            )

            # Log
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
                final_player_hand_size=len(player_hand),
                final_dealer_hand_size=len(dealer_hand),
                final_player_cards=final_player_cards,
                final_dealer_cards=final_dealer_cards
            )
            if hasattr(agent, 'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

        if player_blackjack:
            # Player BJ => immediate 1.5x payoff
            winnings = wager * 1.5
            bankroll += (winnings + wager)
            outcome = 'win'
            final_profit = bankroll - hand_starting_bankroll

            dealer_final_val = environment.calculate_hand_value(dealer_hand)[0]
            player_final_val = environment.calculate_hand_value(player_hand)[0]
            did_player_bust = (player_final_val > 21)
            did_dealer_bust = (dealer_final_val > 21)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in player_hand)
            final_dealer_cards = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(
                deck, running_count, shoes_played
            )

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
                final_player_hand_size=len(player_hand),
                final_dealer_hand_size=len(dealer_hand),
                final_player_cards=final_player_cards,
                final_dealer_cards=final_dealer_cards
            )
            if hasattr(agent, 'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

        # ---------- Normal Flow ----------
        actions_this_hand = []
        dealer_actions    = []

        # For Sarsa, we need to track s, a, and also s' and a'
        last_state  = None
        last_action = None

        while current_hand_index < len(player_hands):
            phand = player_hands[current_hand_index]

            # (s, a) for the current step
            s  = None
            a  = None

            # choose next action *before* applying it, so Sarsa can see next_a
            while True:
                pval, is_soft = environment.calculate_hand_value(phand)
                if pval > 21:  # busted
                    break

                # Build available actions
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
                    # QLearning or Sarsa => only 'hit','stand'
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

                # -------------------------
                # (A) If last_action exists AND RL_METHOD == "Sarsa",
                #     do the Sarsa update from the (last_state, last_action)
                #     to the newly chosen (state, action).
                # -------------------------
                if (
                    config.RL_METHOD == "Sarsa"
                    and last_state is not None
                    and last_action is not None
                ):
                    # We haven't computed reward yet because the last step's outcome
                    # was not final. So in a real Sarsa, you'd accumulate rewards for
                    # each "step" to handle partial reward (or 0 reward) for a "hit."
                    #
                    # Often we do step-by-step rewards 0 for each "hit" and final
                    # reward at the end. For simplicity, let's do 0 for intermediate
                    # hits:
                    step_reward = 0.0  # no immediate outcome
                    agent.update(last_state, last_action, step_reward, state, action)

                # Update last_state, last_action for next step
                last_state  = state
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
                    can_split_this = (
                        environment.can_split(phand)
                        and splits_done < config.MAX_SPLITS
                        and bankroll >= wagers[current_hand_index]
                    )
                    if can_split_this:
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
                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        phand.append(card)
                        dealt_cards_this_shoe.append(card)
                        pval, is_soft = environment.calculate_hand_value(phand)
                        if pval > 21:
                            break

            current_hand_index += 1

        # Dealer logic: only play if at least one player hand not bust
        dealer_must_play = any(environment.calculate_hand_value(h)[0] <= 21 for h in player_hands)
        if dealer_must_play:
            dealer_hand_copy = dealer_hand.copy()
            dealer_hand_copy, dealer_value, running_count, dealer_hits = environment.dealer_turn(
                dealer_hand_copy, deck, running_count
            )
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

            # Final RL update (Q-Learning or Sarsa):
            # Terminal next_state => (0,0,0,0)
            # In Sarsa, next_action = None in a terminal
            if config.RL_METHOD in ["QLearning","Sarsa"] and last_action is not None and last_state is not None:
                if outcome == 'win':
                    reward = 1.0
                elif outcome == 'push':
                    reward = 0.0
                else:
                    reward = -1.0

                next_state = (0,0,0,0)

                if config.RL_METHOD == "QLearning":
                    agent.update(last_state, last_action, reward, next_state)
                else:
                    # True Sarsa: terminal => next_action=None
                    next_action = None
                    agent.update(last_state, last_action, reward, next_state, next_action)

            # Logging final states
            dealer_final_val = dealer_value
            player_final_val = environment.calculate_hand_value(phand)[0]
            did_player_bust  = (player_final_val > 21)
            did_dealer_bust  = (dealer_final_val > 21)

            if dealer_must_play:
                final_dealer_hand_size = len(dealer_hand_copy)
                final_dealer_cards     = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand_copy)
            else:
                final_dealer_hand_size = len(dealer_hand)
                final_dealer_cards     = "|".join(f"{c[0]}-{c[1]}" for c in dealer_hand)

            final_player_cards = "|".join(f"{c[0]}-{c[1]}" for c in phand)

            logger.log_hand(
                outcome=outcome,
                profit=final_profit,
                bankroll=bankroll,
                actions_taken=actions_this_hand,
                starting_true_count=start_of_hand_tc,
                starting_decks_remaining=start_decks_remaining,
                dealer_actions=dealer_actions,
                dealer_final_total=dealer_value,
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
                final_player_hand_size=len(phand),
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

    # Done with main loop
    summary = analytics.print_summary(logger, total_deals=hands_played)

    # Single-run plots
    if logger.get_bankroll_history():
        analytics.plot_bankroll_over_time(logger.get_bankroll_history())
    analytics.plot_ev_over_time(logger)
    if config.RL_METHOD in ["QLearning","Sarsa"]:
        analytics.plot_epsilon_convergence(logger)

    # Restore config
    config.NUM_SHOES_TO_PLAY = original_num_shoes

    return {
        "hands_played": hands_played,
        "shoes_played": shoes_played,
        "summary": summary
    }, agent, logger

def main():
    """
    Simple entry point if someone does: python main.py
    """
    return run_main_simulation(agent_override=None, num_shoes=None)