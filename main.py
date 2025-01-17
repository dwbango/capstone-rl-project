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

    # RL agents: hit/stand only; BasicStrategy: hit/stand/double/split
    ACTIONS_RL = ['hit','stand']
    ACTIONS_BS = ['hit','stand','double','split']

    agent = create_agent(ACTIONS_RL if config.RL_METHOD != "BasicStrategy" else ACTIONS_BS)

    running_count = 0
    shoes_played = 0
    hands_played = 0
    splits_done = 0

    deck = environment.shuffle_deck(environment.create_deck())
    dealt_cards_this_shoe = []

    while shoes_played < config.NUM_SHOES_TO_PLAY and bankroll > 0:
        hand_starting_bankroll = bankroll
        try:
            bankroll, wager = strategy.place_wager(bankroll)
        except ValueError:
            break

        player_hand = []
        dealer_hand = []

        # Deal initial cards
        for _ in range(2):
            card, running_count, true_count_int = environment.deal_card(deck, running_count)
            player_hand.append(card)
            dealt_cards_this_shoe.append(card)

        for _ in range(2):
            card, running_count, true_count_int = environment.deal_card(deck, running_count)
            dealer_hand.append(card)
            dealt_cards_this_shoe.append(card)

        player_hands = [player_hand]
        wagers = [wager]
        current_hand_index = 0

        player_value, is_soft = environment.calculate_hand_value(player_hand)
        dealer_value, _ = environment.calculate_hand_value(dealer_hand)
        soft_flag = 1 if is_soft else 0
        starting_true_count = true_count_int
        starting_decks_remaining = len(deck)/52.0

        # Check dealer blackjack
        if environment.is_blackjack(dealer_hand):
            outcome = 'push' if environment.is_blackjack(player_hand) else 'lose'
            bankroll = strategy.update_bankroll(bankroll, wager, outcome)
            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll, [], starting_true_count, starting_decks_remaining, dealer_actions=[])
            if shoes_played > old_shoes_played:
                logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
                dealt_cards_this_shoe = []
            hands_played += 1
            if hasattr(agent,'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

        # Check player blackjack
        if environment.is_blackjack(player_hand):
            winnings = (wager * 1.5)
            bankroll += (winnings + wager)
            outcome = 'win'
            running_count, old_shoes_played = running_count, shoes_played
            running_count, shoes_played = environment.reshuffle_if_needed(deck, running_count, shoes_played)
            final_profit = bankroll - hand_starting_bankroll
            logger.log_hand(outcome, final_profit, bankroll, [], starting_true_count, starting_decks_remaining, dealer_actions=[])
            if shoes_played > old_shoes_played:
                logger.log_shoe_data(shoes_played, dealt_cards_this_shoe)
                dealt_cards_this_shoe = []
            hands_played += 1
            if hasattr(agent,'epsilon'):
                logger.log_epsilon(agent.epsilon)
            continue

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
                    can_double = (len(phand)==2 and bankroll >= wagers[current_hand_index])
                    can_split_hand = (environment.can_split(phand) and splits_done < config.MAX_SPLITS and bankroll >= wagers[current_hand_index])
                    if can_double:
                        available_actions.append('double')
                    if can_split_hand:
                        available_actions.append('split')
                else:
                    available_actions = ['hit','stand']

                state = (pval, dealer_value, 1 if is_soft else 0, true_count_int)
                if config.RL_METHOD == "BasicStrategy":
                    action = agent.choose_action(state, available_actions, phand, dealer_hand, bankroll, wagers[current_hand_index], splits_done)
                else:
                    # RL agents only know hit/stand
                    if 'double' in available_actions:
                        available_actions.remove('double')
                    if 'split' in available_actions:
                        available_actions.remove('split')
                    action = agent.choose_action(state, available_actions)

                actions_this_hand.append(action)
                last_state = state
                last_action = action

                if action == 'hit':
                    card, running_count, true_count_int = environment.deal_card(deck, running_count)
                    phand.append(card)
                    dealt_cards_this_shoe.append(card)
                    pval, is_soft = environment.calculate_hand_value(phand)
                    if pval > 21:
                        break

                elif action == 'stand':
                    break

                elif action == 'double' and config.RL_METHOD == "BasicStrategy":
                    if len(phand)==2 and bankroll>=wagers[current_hand_index]:
                        bankroll -= wagers[current_hand_index]
                        wagers[current_hand_index]*=2
                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        phand.append(card)
                        dealt_cards_this_shoe.append(card)
                    break

                elif action == 'split' and config.RL_METHOD == "BasicStrategy":
                    if environment.can_split(phand) and splits_done<config.MAX_SPLITS and bankroll>=wagers[current_hand_index]:
                        bankroll -= wagers[current_hand_index]
                        c1,c2=phand[0], phand[1]
                        new_hand_1=[c1]
                        new_hand_2=[c2]
                        player_hands[current_hand_index]=new_hand_1
                        player_hands.insert(current_hand_index+1,new_hand_2)
                        wagers.insert(current_hand_index+1,wagers[current_hand_index])
                        splits_done+=1
                        card, running_count, true_count_int = environment.deal_card(deck, running_count)
                        new_hand_1.append(card)
                        dealt_cards_this_shoe.append(card)
                    else:
                        card, running_count, true_count_int = environment.deal_card(deck,running_count)
                        phand.append(card)
                        dealt_cards_this_shoe.append(card)
                        pval, is_soft = environment.calculate_hand_value(phand)
                        if pval>21:
                            break

            current_hand_index+=1

        dealer_must_play = any(environment.calculate_hand_value(h)[0]<=21 for h in player_hands)
        if dealer_must_play:
            dealer_hand_copy=dealer_hand.copy()
            dealer_hand_copy,dealer_value,running_count,dealer_hits=environment.dealer_turn(dealer_hand_copy,deck,running_count)
            if dealer_hits>0:
                dealer_actions=['hit']*dealer_hits
                if dealer_value<=21:
                    dealer_actions.append('stand')
            else:
                dealer_actions=['stand']
        else:
            dealer_actions=['stand']

        # Determine outcomes for each hand
        for i, phand in enumerate(player_hands):
            pval,_=environment.calculate_hand_value(phand)
            w=wagers[i]
            if pval<=21:
                if dealer_value>21 or pval>dealer_value:
                    outcome='win'
                    bankroll=strategy.update_bankroll(bankroll,w,'win')
                elif pval<dealer_value:
                    outcome='lose'
                    bankroll=strategy.update_bankroll(bankroll,w,'lose')
                else:
                    outcome='push'
                    bankroll=strategy.update_bankroll(bankroll,w,'push')
            else:
                outcome='lose'
                bankroll=strategy.update_bankroll(bankroll,w,'lose')

            final_profit=bankroll-hand_starting_bankroll

            # RL rewards and updates
            if config.RL_METHOD in ["QLearning","Sarsa"] and last_action is not None and last_state is not None:
                # Reward scheme: win=+1, push=0, lose=-1
                if outcome == 'win':
                    reward = 1.0
                elif outcome == 'push':
                    reward = 0.0
                else:
                    reward = -1.0

                next_state = (0,0,0,0) # Terminal
                if config.RL_METHOD == "QLearning":
                    # Normal Q-learning terminal update
                    agent.update(last_state, last_action, reward, next_state)
                else:
                    # Sarsa terminal update (no next_action)
                    old_q = agent.get_q_value(last_state, last_action)
                    new_q = old_q + agent.alpha * (reward - old_q)
                    agent.q_table[(last_state, last_action)] = new_q
                    # Manually decay epsilon for Sarsa after terminal
                    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)

            logger.log_hand(outcome,final_profit,bankroll,actions_this_hand,starting_true_count,starting_decks_remaining,dealer_actions)

        running_count, old_shoes_played=running_count,shoes_played
        running_count, shoes_played=environment.reshuffle_if_needed(deck,running_count,shoes_played)
        if shoes_played>old_shoes_played:
            logger.log_shoe_data(shoes_played,dealt_cards_this_shoe)
            dealt_cards_this_shoe=[]

        hands_played+=1
        if hasattr(agent,'epsilon') and config.RL_METHOD == "QLearning":
            # QLearning epsilon decay after each hand already happens in agent.update
            pass

    summary=analytics.print_summary(logger)
    bankroll_history=logger.get_bankroll_history()
    if bankroll_history:
        analytics.plot_bankroll_over_time(bankroll_history)

    if config.RL_METHOD in ["QLearning","Sarsa"]:
        analytics.plot_epsilon_convergence(logger)

    return {"hands_played":hands_played,"shoes_played":shoes_played,"summary":summary},agent,logger

if __name__=="__main__":
    main()

