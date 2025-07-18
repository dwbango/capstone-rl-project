Copyright (c) 2023 Donald Bango
All rights reserved.

This repository is provided solely for viewing and educational purposes.
No permission is granted for use, distribution, modification, or any other 
activity without explicit written consent from the owner.

Reinforcement Learning Evaluation & Statistical Analysis in a Stochastic Blackjack Environment

A comprehensive Blackjack simulation project featuring:
	•	Reinforcement Learning (Q-Learning, SARSA),
	•	Classic Strategies (Basic Strategy, Random),
	•	Count-based or Flat Betting,
	•	Statistical Comparisons (ANOVA),
	•	Detailed Logging & Visualizations.

1. Project Overview

Blackjack is a popular casino card game with a built-in house edge. By tracking high and low-value cards in finite decks (card counting), players can sometimes overcome the house advantage. Meanwhile, Reinforcement Learning (RL), especially Q-Learning (off-policy) and SARSA (on-policy), has emerged as a robust approach to learning optimal actions in dynamic environments.

This project explores how RL agents (Q-Learning, SARSA) compare against conventional approaches (Basic Strategy, Random) in a configurable Blackjack environment. Users can:
	•	Modify game rules (decks, shuffle point, max splits, etc.),
	•	Choose an agent,
	•	Select a betting style (flat or spread),
	•	Run simulations with real-time analytics,
	•	Perform multi-method comparisons,
	•	Run an ANOVA test (requires Redis + RQ) to see if the EV differences are statistically significant.

Note:
	•	CSV reports are generated only for single-run simulations. Multi-method compare or ANOVA results are not saved to CSV but displayed in the UI.
	•	For ANOVA features, you must run Redis locally and start an RQ worker.

2. Key Features
	1.	Flexible Game Rules
	•	Configure the number of decks, shuffle point (deck penetration), and max splits.
	2.	Multiple Agents
	•	Q-Learning (fresh or pretrained),
	•	SARSA (fresh or pretrained),
	•	Basic Strategy,
	•	Random (baseline).
	3.	Betting Options
	•	Flat or Spread (count-based).
	•	Bankroll is clamped to a minimum of 1 and maximum of 100,000.
	4.	Interactive Flask UI
	•	Single-run or Compare All (auto-runs 4 strategies).
	•	Real-time plots (bankroll, EV).
	•	Summaries (final bankroll, win/loss/push rates, variance, etc.).
	5.	Statistical Analysis (Optional)
	•	ANOVA with post-hoc tests (Bonferroni-corrected) to check if differences among strategies are significant.
	•	Requires running Redis + RQ worker locally.
	6.	Detailed Reporting
	•	Single-run simulation can export CSV or ZIP containing summary metrics, hand-by-hand logs, and shoe-level data.
	•	Compare All runs produce combined plots but no CSV export.
	•	RL strategy charts for Q-Learning/SARSA (hard/soft/pairs) are available.

3. Technical Stack & Requirements
	•	Python 3.7+
	•	Flask (web interface)
	•	matplotlib and scipy for plotting and statistical testing
	•	Redis + RQ (only if you want to use ANOVA background jobs)

Install the dependencies from requirements.txt. For example:

Flask==2.2.2
matplotlib==3.6.0
numpy==1.23.4
pandas==1.5.0
scipy==1.9.3
redis==4.3.4
rq==1.10.1

4. Quick Start: Installation & Usage
	1.	Clone the Repository

git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName


	2.	(Optional) Virtual Environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
# or
venv\Scripts\activate      # Windows


	3.	Install Dependencies

pip install -r requirements.txt


	4.	Run the Flask App

python app.py

Visit http://127.0.0.1:5000.

	5.	(Optional) Use ANOVA / Repeated Sims
	•	Ensure Redis is running locally (redis-server).
	•	Start an RQ worker in another terminal:

rq worker --url=redis://localhost:6379


	•	Then, in the UI, click Run Stats (ANOVA) to queue the background job.

5. Project Structure
```text
.
├── app.py                      # Flask app & routes
├── analytics.py                # DataLogger & plotting
├── rl_agent.py                 # RL agent classes (Q-Learning, SARSA, etc.)
├── strategy.py                 # Betting logic (flat/spread), bankroll updates
├── environment/                # Deck creation, dealing, counting
├── templates/                  # HTML templates for Flask
├── static/                     # CSS, JS, and generated plot images
├── tasks.py                    # RQ tasks for background computations (ANOVA)
├── user_agents/                # Saved (pickled) RL agents
├── config.py                   # Global config (STARTING_BANKROLL, etc.)
├── main.py                     # Orchestrates environment + RL agent
└── README.md                   # This file

6. Detailed Usage
	1.	Single Run Simulation
	•	Set parameters (num. decks, shuffle point, max splits).
	•	Choose an agent (Q-Learning, SARSA, Basic Strategy, etc.).
	•	If Q-Learning or SARSA is fresh, set alpha/gamma/epsilon/decay.
	•	If Pretrained or UserAgent_, it uses a saved .pkl agent.
	•	Select flat or spread betting, specify initial bankroll.
	•	Click “Run Simulation.”
	•	Plots (bankroll vs. hand, EV vs. hand) refresh, plus a summary.
	•	Reports: You can Generate Report (Single CSV) or Download All CSVs (zip) to see final metrics & logs.
	•	Note: CSVs are for single-run only.
	2.	Compare All Methods
	•	Same form parameters, but click “Compare Player Agents.”
	•	Internally runs 4 strategies: Basic Strategy, Random, Q-Learning, SARSA.
	•	Returns combined plots of bankroll & EV over time.
	•	No CSV export for multi-method compares.
	3.	ANOVA (Repeated Sims)
	•	Requires local Redis + RQ worker.
	•	Click “Run Stats (ANOVA)” to enqueue repeated runs for each method.
	•	The job is processed in the background; results appear in the UI once done.
	•	No CSV for ANOVA; results are displayed in the UI (F-stat, p-value, pairwise t-tests).
	4.	Save Agent (Q-Learning/SARSA)
	•	After a run, you can save the RL agent by naming it.
	•	This pickle file is stored in user_agents/.
	•	Next time, choose UserAgent_<Name> to load it with epsilon=0 (fully greedy).

7. Data Outputs & Visualizations
	•	Single-Run Plots:
	•	bankroll_history.png – Bankroll vs. Hand #
	•	ev_over_time.png – Cumulative average profit/hand vs. Hand #
	•	epsilon_convergence.png – If RL used, plots epsilon decaying over hands.
	•	Multi-Method Compare:
	•	compare_bankroll.png – Combined bankroll lines for each method
	•	ev_compare.png – Combined EV lines for each method
	•	RL Strategy Charts:
	•	strategy_chart_hard.png, strategy_chart_soft.png, strategy_chart_pairs.png – Heatmaps showing Q-Learning/SARSA decisions for different states.

Interpreting the Charts
	•	Bankroll vs. Hand #: Trend lines for net gains or losses.
	•	EV (Profit/Hand) vs. Hand #: Indicates the average win/loss per hand over time.
	•	Variance: If final logs show large variance, it implies more volatile swings in the bankroll.
	•	Epsilon Convergence: Low epsilon near zero means the agent is mostly exploiting its learned Q-values (converged policy). A slow or partial decline can mean ongoing exploration.

8. Reinforcement Learning Details
	1.	Q-LearningAgent
	•	Off-policy updates from max-Q over actions, uses alpha/gamma/epsilon.
	•	State typically includes (player total, dealer upcard, is_soft, true_count).
	2.	SARSAAgent
	•	On-policy updates from actual next action in the trajectory.
	•	Also uses alpha/gamma/epsilon but with the SARSA formula.
	3.	Basic StrategyAgent
	•	Hard-coded Blackjack Basic Strategy, ignoring card counting.
	4.	RandomAgent
	•	A baseline that randomly picks from available actions.

9. Statistical Analysis
	1.	ANOVA
	•	Conducted on repeated runs for each method (Random, Basic Strategy, Q-Learning, SARSA) to compare their EV distributions.
	•	If p < 0.05 with multiple methods, pairwise t-tests (Bonferroni-corrected) highlight which pairs differ significantly.
	•	Must have Redis + RQ set up locally (not provided by default).
	2.	Confidence Intervals
	•	95% CIs around EV for each method show variability in average outcomes.

Note: The ANOVA and post-hoc results are displayed in the UI, not exported to CSV.

10. Future Enhancements & Contributing
	1.	Potential Extensions
	•	Use deeper RL approaches (Deep Q-Networks, etc.).
	•	Expand the environment with side bets or multiple-player scenarios.
	•	Animate card dealing for a more interactive front-end experience.
	2.	Contributions
	•	Originally built as a data science capstone project.
	•	Forks, PRs, and issue reports are welcome. Provide feedback or propose improvements!

11. License
Copyright (c) 2023 Donald Bango
All rights reserved.

This repository is provided solely for viewing and educational purposes.
No permission is granted for use, distribution, modification, or any other 
activity without explicit written consent from the owner.

Contact / Further Info
	•	Author: Donald Bango
	•	Email: donaldbango11@aol.com
	•	GitHub: dwbango

For any questions, bug reports, or suggestions, please open an issue or pull request and email the developer.

Thank you for exploring this Blackjack RL & Statistical Analysis tool! We hope it aids in understanding how RL and statistical methods apply to Blackjack strategy experimentation.

(End of README)
