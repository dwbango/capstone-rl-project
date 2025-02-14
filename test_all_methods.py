# test_all_methods.py

import config
import pickle
from main import run_main_simulation

def run_evaluation(agent_override, method_label, num_shoes):
    """
    Runs run_main_simulation() for 'num_shoes' using either:
      - agent_override (if not None), or 
      - a newly created agent (based on config.RL_METHOD).
    Then prints & returns the summary dict.
    """
    results, final_agent, _ = run_main_simulation(agent_override=agent_override, num_shoes=num_shoes)
    summary = results["summary"]

    print(f"\n--- {method_label} Results ({num_shoes} shoes) ---")
    print(f"  Net Profit:  {summary['net_profit']}")
    print(f"  EV/Hand:     {summary['EV_per_hand']:.4f}")
    print(f"  StdDev/Hand: {summary['std_dev']:.4f}")
    print(f"  Win: {summary['win_rate']:.2%}, Loss: {summary['loss_rate']:.2%}, Push: {summary['push_rate']:.2%}")

    return summary

def main():
    test_shoes = 2000  # e.g. your final test run for each method

    # 1) BasicStrategy
    config.RL_METHOD = "BasicStrategy"
    bs_summary = run_evaluation(agent_override=None, method_label="BasicStrategy", num_shoes=test_shoes)

    # 2) Random
    config.RL_METHOD = "Random"
    rand_summary = run_evaluation(agent_override=None, method_label="Random", num_shoes=test_shoes)

    # 3) QLearning (greedy)
    with open("trained_qlearning.pkl","rb") as f:
        ql_agent = pickle.load(f)
    # Force epsilon=0 for purely greedy
    ql_agent.epsilon = 0.0
    ql_agent.epsilon_decay = 1.0

    config.RL_METHOD = "QLearning"
    ql_summary = run_evaluation(agent_override=ql_agent, method_label="QLearning (Greedy)", num_shoes=test_shoes)

    # 4) Sarsa (greedy)
    with open("trained_sarsa.pkl","rb") as f:
        sarsa_agent = pickle.load(f)
    sarsa_agent.epsilon = 0.0
    sarsa_agent.epsilon_decay = 1.0

    config.RL_METHOD = "Sarsa"
    sarsa_summary = run_evaluation(agent_override=sarsa_agent, method_label="Sarsa (Greedy)", num_shoes=test_shoes)

    # Example final table
    print("\n===== Final Comparison Table =====")
    print("Method              | Net Profit |  EV/Hand  |  StDev  | Win%   | Loss%  | Push% ")
    print("--------------------------------------------------------------------------------")
    def row_str(label, s):
        return (f"{label:<20} | "
                f"{s['net_profit']:>10,.0f} | "
                f"{s['EV_per_hand']:.4f} | "
                f"{s['std_dev']:.4f} | "
                f"{s['win_rate']:.2%} | "
                f"{s['loss_rate']:.2%} | "
                f"{s['push_rate']:.2%}")
    print(row_str("BasicStrategy", bs_summary))
    print(row_str("Random", rand_summary))
    print(row_str("QLearning(G)", ql_summary))
    print(row_str("Sarsa(G)", sarsa_summary))

if __name__ == "__main__":
    main()