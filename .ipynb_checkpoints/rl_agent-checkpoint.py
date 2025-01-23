# rl_agent.py

import random
import config
import strategy

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, actions, alpha=0.06, gamma=0.95, epsilon=1.0, epsilon_decay=0.99999):
        """
        actions: list of possible actions (e.g., ['hit', 'stand'])
        alpha: learning rate (0 < alpha <= 1)
        gamma: discount factor (0 <= gamma <= 1)
        epsilon: initial exploration rate (0 <= epsilon <= 1)
        epsilon_decay: factor by which epsilon is multiplied after each update 
                       to gradually reduce exploration.
                       For example, epsilon_decay=0.999 means epsilon * 0.999 each step.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Keys: (state, action), Value: Q-value (float)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        """
        Epsilon-greedy action selection.
        With probability epsilon, choose a random available action.
        Otherwise, choose the action with the highest Q-value at this state.
        """
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [(self.get_q_value(state, a), a) for a in available_actions]
            _, best_action = max(q_values, key=lambda x: x[0])
            return best_action

    def update(self, state, action, reward, next_state):
        """
        Q-learning update:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
        """
        old_q = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        best_next_q = max(next_q_values) if next_q_values else 0.0

        new_q = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
        self.q_table[(state, action)] = new_q

        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

# Sarsa Agent
class SarsaAgent:
    def __init__(self, actions, alpha=0.06, gamma=0.95, epsilon=1.0, epsilon_decay=0.99999):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [(self.get_q_value(state, a), a) for a in available_actions]
            _, best_action = max(q_values, key=lambda x: x[0])
            return best_action

    def update(self, state, action, reward, next_state, next_action):
        """
        Sarsa update rule:
        Q(s,a) = Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
        where a' is the action chosen in next_state.
        """
        old_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        new_q = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
        self.q_table[(state, action)] = new_q

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

# Basic Strategy Agent (no learning, just a policy)
class BasicStrategyAgent:
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state, available_actions, player_hand, dealer_hand, bankroll, current_wager, splits_done=0):
        # Determine if double/split is possible
        can_double = (len(player_hand) == 2 and bankroll >= current_wager)
        can_split_hand = strategy.is_pair(player_hand)
        dealer_card = dealer_hand[0]

        action = strategy.get_player_decision(
            hand=player_hand,
            dealer_card=dealer_card,
            bankroll=bankroll,
            current_wager=current_wager,
            can_double=can_double,
            can_split_hand=can_split_hand,
            splits_done=splits_done,
            max_splits=config.MAX_SPLITS
        )

        # Ensure chosen action is in available_actions
        if action not in available_actions:
            if 'hit' in available_actions:
                action = 'hit'
            else:
                action = available_actions[0]

        return action

