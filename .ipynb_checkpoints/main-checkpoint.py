# main.py
import config
import environment
import strategy
import analytics
from rl_agent import QLearningAgent, SarsaAgent, BasicStrategyAgent

def create_agent(actions):
    if config.RL_METHOD == "QLearning":
        return QLearningAgent(actions)
    elif config.RL_METHOD == "Sarsa":
        return SarsaAgent(actions)
    elif config.RL_METHOD == "BasicStrategy":
        return BasicStrategyAgent(actions)
    else:
        raise ValueError(f"Unknown RL method: {config.RL_METHOD}")

def main():
    logger = analytics.DataLogger()
    bankroll = strategy.initialize_bankroll()
    deck = environment.shuffle_deck(environment.create_deck())

    ACTIONS = ['hit', 'stand', 'double']
    agent = create_agent(ACTIONS)  # Agent chosen based on config.RL_METHOD

    running_count = 0
    shoes_played = 0
    hands_played = 0
    splits_done = 0  # Track splits if needed (0 by default)

    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:
        hand_starting_bankroll = bankroll

        try:
            bankroll, wager = strategy.place_wager(bankroll)
        except ValueError:
            break

        player_hand, dealer_hand, running_count, true_count_int = environment.deal_initial_hands(deck, running_count)
        player_value, is_soft = environment.calculate_hand_value(player_hand)
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)
        soft_flag = 1 if is_soft else 0

        # State includes soft_flag and true_count_int
        state = (player_value, dealer_value, soft_flag, true_count_int)

        # Check dealer blackjack
        if environment.is_blackjack(dealer_hand):
            outcome = 'push' if environment.is_blackjack(player_hand) else 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, outcome)
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll)
            hands_played += 1
            continue

        # Check player blackjack
        if environment.is_blackjack(player_hand):
            winnings = (wager * 1.5)
            bankroll += (winnings + wager)
            outcome = 'win'
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll)
            hands_played += 1
            continue

        last_state = state
        last_action = None

        # Player decision loop
        while True:
            available_actions = ['hit', 'stand']
            if len(player_hand) == 2 and bankroll >= wager:
                available_actions.append('double')

            # If BasicStrategy, pass extra params. If RL, just use state & available_actions.
            if config.RL_METHOD == "BasicStrategy":
                action = agent.choose_action(
                    state,
                    available_actions,
                    player_hand,
                    dealer_hand,
                    bankroll,
                    wager,
                    splits_done
                )
            else:
                action = agent.choose_action(state, available_actions)

            if action == 'hit':
                card, running_count, true_count_int = environment.deal_card(deck, running_count)
                player_hand.append(card)
                player_value, is_soft = environment.calculate_hand_value(player_hand)
                soft_flag = 1 if is_soft else 0
                state = (player_value, dealer_value, soft_flag, true_count_int)
                if player_value > 21:
                    break

            elif action == 'stand':
                break

            elif action == 'double':
                if len(player_hand) == 2 and bankroll >= wager:
                    bankroll -= wager
                    wager *= 2
                    card, running_count, true_count_int = environment.deal_card(deck, running_count)
                    player_hand.append(card)
                    player_value, is_soft = environment.calculate_hand_value(player_hand)
                    soft_flag = 1 if is_soft else 0
                    state = (player_value, dealer_value, soft_flag, true_count_int)
                    break
                else:
                    # If can't double, treat as hit
                    card, running_count, true_count_int = environment.deal_card(deck, running_count)
                    player_hand.append(card)
                    player_value, is_soft = environment.calculate_hand_value(player_hand)
                    soft_flag = 1 if is_soft else 0
                    state = (player_value, dealer_value, soft_flag, true_count_int)
                    if player_value > 21:
                        break

            last_state = state
            last_action = action

        # Dealer turn if player not busted
        if player_value <= 21:
            dealer_hand_copy = dealer_hand.copy()
            dealer_hand_copy, dealer_value, running_count = environment.dealer_turn(dealer_hand_copy, deck, running_count)
        else:
            dealer_value, _ = environment.calculate_hand_value(dealer_hand)

        # Determine outcome
        if player_value <= 21:
            if dealer_value > 21 or player_value > dealer_value:
                outcome = 'win'
                bankroll = strategy.update_bankroll(bankroll, wager, 'win')
            elif player_value < dealer_value:
                outcome = 'lose'
                bankroll = strategy.update_bankroll(bankroll, wager, 'lose')
            else:
                outcome = 'push'
                bankroll = strategy.update_bankroll(bankroll, wager, 'push')
        else:
            outcome = 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, 'lose')

        running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
        final_profit = bankroll - hand_starting_bankroll
        logger.log_hand(outcome, final_profit, bankroll)

        # RL Updates only if QLearning or Sarsa
        if last_action is not None and config.RL_METHOD != "BasicStrategy":
            if config.RL_METHOD == "Sarsa":
                # For Sarsa, we need the next_action
                next_available_actions = ['hit', 'stand']
                if len(player_hand) == 2 and bankroll >= wager:
                    next_available_actions.append('double')
                next_action = agent.choose_action(state, next_available_actions)
                agent.update(last_state, last_action, final_profit, state, next_action)
            else:
                # Q-Learning doesn't need next_action
                agent.update(last_state, last_action, final_profit, state)

        hands_played += 1

    # Print summary (via analytics)
    summary = analytics.print_summary(logger)

    bankroll_history = logger.get_bankroll_history()
    if bankroll_history:
        analytics.plot_bankroll_over_time(bankroll_history)

    results = {
        "hands_played": hands_played,
        "shoes_played": shoes_played
    }
    return results

if __name__ == "__main__":
    main()


