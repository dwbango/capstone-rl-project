# tasks.py
import pickle
import config
from main import run_main_simulation
import analytics

def run_anova_background(repeats, shoes_per_run):
    """
    Heavy-lifting logic for repeated runs & ANOVA.
    Called by RQ as a background job.
    """
    methods = ["BasicStrategy", "Random", "QLearning", "Sarsa"]
    ev_data = {m: [] for m in methods}

    for m in methods:
        agent_override = None
        if m == "QLearning":
            try:
                with open("trained_qlearning.pkl","rb") as f:
                    agent_override = pickle.load(f)
                agent_override.epsilon = 0.0
                agent_override.epsilon_decay = 1.0
            except FileNotFoundError:
                pass
        elif m == "Sarsa":
            try:
                with open("trained_sarsa.pkl","rb") as f:
                    agent_override = pickle.load(f)
                agent_override.epsilon = 0.0
                agent_override.epsilon_decay = 1.0
            except FileNotFoundError:
                pass

        old_shoes = config.NUM_SHOES_TO_PLAY
        config.RL_METHOD = m

        for _ in range(repeats):
            config.NUM_SHOES_TO_PLAY = shoes_per_run
            results, _, _ = run_main_simulation(agent_override=agent_override)
            final_ev = results["summary"]["EV_per_hand"]
            ev_data[m].append(final_ev)

        # restore original config
        config.NUM_SHOES_TO_PLAY = old_shoes

    # Perform ANOVA
    anova_results = analytics.run_anova_and_posthoc(ev_data)

    # Calculate means
    mean_evs = {}
    for m in methods:
        if ev_data[m]:
            mean_evs[m] = sum(ev_data[m]) / len(ev_data[m])
        else:
            mean_evs[m] = None

    # Confidence intervals, if available
    method_stats = {}
    if hasattr(analytics, 'compute_confidence_intervals'):
        method_stats = analytics.compute_confidence_intervals(ev_data)

    # Convert numpy.bool_ -> bool
    if "pairwise" in anova_results:
        for pair in anova_results["pairwise"]:
            if "significant_after_bonf" in pair:
                pair["significant_after_bonf"] = bool(pair["significant_after_bonf"])

    # Return dict -> stored in job.result
    return {
        "mean_evs": mean_evs,
        "anova_f": anova_results.get("anova_f"),
        "anova_p": anova_results.get("anova_p"),
        "pairwise": anova_results.get("pairwise", []),
        "error": anova_results.get("error", ""),
        "method_stats": method_stats,
        "repeats_used": repeats,
        "shoes_per_run_used": shoes_per_run
    }