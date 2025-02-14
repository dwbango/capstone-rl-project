# train_qlearning.py

import config
import pickle
from main import run_main_simulation

def train_qlearning():
    # 1) Force the training config you want:
    config.RL_METHOD = "QLearning"
    config.NUM_SHOES_TO_PLAY = 100000
    config.NUM_DECKS = 2
    config.SHUFFLE_POINT = 0.25
    config.MAX_SPLITS = 3
    config.BETTING_STYLE = "flat"
    config.DEFAULT_WAGER = 10
    # config.STARTING_BANKROLL = 500000

    # 2) Run the simulation
    results, trained_agent, logger = run_main_simulation()

    # 3) Print final stats
    summary = results["summary"]
    print("=== Q-Learning Training Complete ===")
    print(f"Hands: {results['hands_played']}, Shoes: {results['shoes_played']}")
    print(f"Final Bankroll: {summary['final_bankroll']}, Net Profit: {summary['net_profit']}")
    print(f"EV per hand: {summary['EV_per_hand']:.4f}, Std Dev: {summary['std_dev']:.4f}")
    print(f"Win%: {summary['win_rate']:.2%}, Loss%: {summary['loss_rate']:.2%}, Push%: {summary['push_rate']:.2%}")

    # 4) Pickle the agent
    with open("trained_qlearning.pkl", "wb") as f:
        pickle.dump(trained_agent, f)
    print("Saved Q-Learning agent as 'trained_qlearning.pkl'")

if __name__ == "__main__":
    train_qlearning()