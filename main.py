# main.py
import config
import environment
import strategy
import analytics
from rl_agent import QLearningAgent  # Using the RL agent

def main():
    logger = analytics.DataLogger()
    bankroll = strategy.initialize_bankroll()
    deck = environment.shuffle_deck(environment.create_deck())

    # Now including 'double' in actions
    ACTIONS = ['hit', 'stand', 'double']
    agent = QLearningAgent(ACTIONS)

    running_count = 0
    shoes_played = 0
    hands_played = 0

    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:
        hand_starting_bankroll = bankroll

        try:
            bankroll, wager = strategy.place_wager(bankroll)
        except ValueError:
            # Can't afford a wager, end simulation
            break

        player_hand, dealer_hand, running_count = environment.deal_initial_hands(deck, running_count)
        player_value, _ = environment.calculate_hand_value(player_hand)
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)

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

        # Create initial state
        # For now, still (player_value, dealer_value)
        state = (player_value, dealer_value)

        player_hands = [player_hand]
        wagers = [wager]

        last_state = state
        last_action = None

        # Player decision loop
        while True:
            # Determine available actions:
            available_actions = ['hit', 'stand']
            # Double allowed if 2 cards and bankroll >= wager
            if len(player_hand) == 2 and bankroll >= wagers[0]:
                available_actions.append('double')

            action = agent.choose_action(state, available_actions)

            if action == 'hit':
                card, running_count = environment.deal_card(deck, running_count)
                player_hand.append(card)
                player_value, _ = environment.calculate_hand_value(player_hand)
                state = (player_value, dealer_value)
                if player_value > 21:
                    # Busted
                    break

            elif action == 'stand':
                # Player stands
                break

            elif action == 'double':
                # Double down logic
                bankroll -= wagers[0]
                wagers[0] *= 2
                card, running_count = environment.deal_card(deck, running_count)
                player_hand.append(card)
                player_value, _ = environment.calculate_hand_value(player_hand)
                state = (player_value, dealer_value)
                # Turn ends after double
                break

            last_state = state
            last_action = action

        # Dealer turn if player hasn't busted
        if any(environment.calculate_hand_value(h)[0] <= 21 for h in player_hands):
            dealer_hand_copy = dealer_hand.copy()
            dealer_hand_copy, dealer_value, running_count = environment.dealer_turn(dealer_hand_copy, deck, running_count)
        else:
            dealer_value, _ = environment.calculate_hand_value(dealer_hand)

        # Determine outcomes
        subhand_outcomes = []
        for i, hand in enumerate(player_hands, start=1):
            hand_value, _ = environment.calculate_hand_value(hand)
            if hand_value <= 21:
                if dealer_value > 21 or hand_value > dealer_value:
                    subhand_outcomes.append('win')
                    bankroll = strategy.update_bankroll(bankroll, wagers[i-1], 'win')
                elif hand_value < dealer_value:
                    subhand_outcomes.append('lose')
                    bankroll = strategy.update_bankroll(bankroll, wagers[i-1], 'lose')
                else:
                    subhand_outcomes.append('push')
                    bankroll = strategy.update_bankroll(bankroll, wagers[i-1], 'push')
            else:
                subhand_outcomes.append('lose')
                bankroll = strategy.update_bankroll(bankroll, wagers[i-1], 'lose')

        running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)

        if len(subhand_outcomes) == 1:
            final_outcome = subhand_outcomes[0]
        else:
            unique_outcomes = set(subhand_outcomes)
            final_outcome = subhand_outcomes[0] if len(unique_outcomes) == 1 else 'mixed'

        final_profit = bankroll - hand_starting_bankroll
        logger.log_hand(final_outcome, final_profit, bankroll)

        # Update Q-values with final_profit as reward
        next_state = state
        if last_action is not None:
            agent.update(last_state, last_action, final_profit, next_state)

        hands_played += 1

    analytics.print_summary(logger)
    bankroll_history = logger.get_bankroll_history()
    if bankroll_history:
        analytics.plot_bankroll_over_time(bankroll_history)

    print(f"Total Hands Played: {hands_played}")
    print(f"Shoes Played: {shoes_played}")

if __name__ == "__main__":
    main()
