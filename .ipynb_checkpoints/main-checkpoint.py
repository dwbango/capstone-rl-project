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

    ACTIONS = ['hit', 'stand', 'double']
    agent = create_agent(ACTIONS)

    running_count = 0
    shoes_played = 0
    hands_played = 0
    splits_done = 0

    deck = environment.shuffle_deck(environment.create_deck())
    dealt_cards_this_shoe = []  # Track all dealt cards in current shoe

    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:
        hand_starting_bankroll = bankroll

        try:
            bankroll, wager = strategy.place_wager(bankroll)
        except ValueError:
            break

        # Deal initial cards individually to log them
        player_hand = []
        dealer_hand = []

        # Player first card
        card, running_count, true_count_int = environment.deal_card(deck, running_count)
        player_hand.append(card)
        dealt_cards_this_shoe.append(card)

        # Player second card
        card, running_count, true_count_int = environment.deal_card(deck, running_count)
        player_hand.append(card)
        dealt_cards_this_shoe.append(card)

        # Dealer first card
        card, running_count, true_count_int = environment.deal_card(deck, running_count)
        dealer_hand.append(card)
        dealt_cards_this_shoe.append(card)

        # Dealer second card
        card, running_count, true_count_int = environment.deal_card(deck, running_count)
        dealer_hand.append(card)
        dealt_cards_this_shoe.append(card)

        player_value, is_soft = environment.calculate_hand_value(player_hand)
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)
        soft_flag = 1 if is_soft else 0
        starting_true_count = true_count_int
        starting_decks_remaining = len(deck) / 52.0
        state = (player_value, dealer_value, soft_flag, true_count_int)

        # Check dealer blackjack
        if environment.is_blackjack(dealer_hand):
            outcome = 'push' if environment.is_blackjack(player_hand) else 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, outcome)
            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll, actions_taken=[], starting_true_count=starting_true_count,
                            starting_decks_remaining=starting_decks_remaining, dealer_actions=[])
            if shoes_played > old_shoes_played:
                logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
                dealt_cards_this_shoe = []
            hands_played += 1
            continue

        # Check player blackjack
        if environment.is_blackjack(player_hand):
            winnings = (wager * 1.5)
            bankroll += (winnings + wager)
            outcome = 'win'
            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll, actions_taken=[], starting_true_count=starting_true_count,
                            starting_decks_remaining=starting_decks_remaining, dealer_actions=[])
            if shoes_played > old_shoes_played:
                logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
                dealt_cards_this_shoe = []
            hands_played += 1
            continue

        last_state = state
        last_action = None
        actions_this_hand = []

        # Player decision loop
        while True:
            available_actions = ['hit', 'stand']
            if len(player_hand) == 2 and bankroll >= wager:
                available_actions.append('double')

            if config.RL_METHOD == "BasicStrategy":
                action = agent.choose_action(state, available_actions, player_hand, dealer_hand, bankroll, wager, splits_done)
            else:
                action = agent.choose_action(state, available_actions)

            actions_this_hand.append(action)

            if action == 'hit':
                card, running_count, true_count_int = environment.deal_card(deck, running_count)
                player_hand.append(card)
                dealt_cards_this_shoe.append(card)
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
                    dealt_cards_this_shoe.append(card)
                    player_value, is_soft = environment.calculate_hand_value(player_hand)
                    soft_flag = 1 if is_soft else 0
                    state = (player_value, dealer_value, soft_flag, true_count_int)
                    break
                else:
                    card, running_count, true_count_int = environment.deal_card(deck, running_count)
                    player_hand.append(card)
                    dealt_cards_this_shoe.append(card)
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
            dealer_hand_copy, dealer_value, running_count, dealer_hits = environment.dealer_turn(dealer_hand_copy, deck, running_count)
            if dealer_hits > 0:
                dealer_actions = ['hit']*dealer_hits
                if dealer_value <= 21:
                    dealer_actions.append('stand')
                else:
                    # If dealer busts after hits, no final stand is needed
                    pass
            else:
                dealer_actions = ['stand']
        else:
            dealer_actions = ['stand']  # Player busted, dealer no hits

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

        running_count, old_shoes_played = running_count, shoes_played
        running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
        final_profit = bankroll - hand_starting_bankroll

        logger.log_hand(outcome, final_profit, bankroll,
                        actions_this_hand,
                        starting_true_count,
                        starting_decks_remaining,
                        dealer_actions)

        if shoes_played > old_shoes_played:
            logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
            dealt_cards_this_shoe = []

        hands_played += 1

    summary = analytics.print_summary(logger)
    bankroll_history = logger.get_bankroll_history()
    if bankroll_history:
        analytics.plot_bankroll_over_time(bankroll_history)

    results = {
        "hands_played": hands_played,
        "shoes_played": shoes_played,
        "summary": summary
    }
    return results

if __name__ == "__main__":
    main()



    
    
    
    
    
    
    
    
    

# -----------------------------
# FUTURE ENHANCEMENTS & NOTES
# -----------------------------
# 1. Deck Logging After Shuffle:
#    - After each reshuffle in environment.py, log or print the new deck order if verbose is True.
#    - This provides insight into the card distribution after each shuffle.

# 2. Verbose Player Actions:
#    - When verbose is True, also log player (agent or basic strategy) actions chosen each turn, 
#      not just the dealer actions.
#    - Add a config.log_message call in main.py right after determining the player's action.

# 3. Risk of Ruin:
#    - Track how often bankroll hits 0 (or below).
#    - At the end of the simulation, calculate the probability (times ruin occurred / total simulations).
#    - Display this metric in the final summary.

# 4. Agent Training Tracking (For RL):
#    - Periodically log agent metrics: current epsilon, average Q-value, cumulative rewards.
#    - Plot or print these metrics to observe learning progress over time.

# 5. Enhanced Analytics & Charts:
#    - Beyond bankroll history, consider plotting win/loss/push rates over time, EV moving averages.
#    - For RL, plot epsilon decay and Q-value trends.
#    - Use DataLogger or a new logger to store these metrics and matplotlib to plot.

# 6. Strategy Chart Generation:
#    - Use get_player_decision to systematically generate a basic strategy chart.
#    - Iterate over possible player totals and dealer cards, record the suggested action.
#    - Print or save this chart as a reference, possibly integrate into the UI.

# future improvements + improving the RL agents and testing baseline training/masking