# rl_agent.py
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999):
        """
        actions: list of possible actions (e.g., ['hit', 'stand'])
        alpha: learning rate
        gamma: discount factor
        epsilon: initial exploration rate
        epsilon_decay: factor by which epsilon is multiplied after each update 
                       to gradually reduce exploration
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Keys: (state, action), Value: Q-value (float)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        With probability epsilon, choose a random action.
        Otherwise, choose the action with the highest Q-value for this state.
        """
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.actions)
        else:
            # Exploit: choose action with max Q-value
            q_values = [(self.get_q_value(state, a), a) for a in self.actions]
            # Find the action with the highest Q-value
            _, best_action = max(q_values, key=lambda x: x[0])
            return best_action

    def update(self, state, action, reward, next_state):
        """
        Q-learning update:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        old_q = self.get_q_value(state, action)

        # Find max Q for next state
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        best_next_q = max(next_q_values) if next_q_values else 0.0

        # Compute new Q-value
        new_q = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
        self.q_table[(state, action)] = new_q

        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
