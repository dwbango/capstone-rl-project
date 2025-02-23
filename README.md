Reinforcement Learning Evaluation & Statistical Analysis in an Stochastic Blackjack Environment

A comprehensive Blackjack simulation product with Reinforcement Learning (RL) capabilities, classical strategies 
(Basic Strategy, Random play), counting-based betting, and detailed analytics. Developed as a data science 
project focusing on statistical comparisons, learning algorithms, and visualization of outcomes in Blackjack.

Table of Contents
	1.	Project Overview
	2.	Key Features
	3.	Technical Stack & Requirements
	4.	Quick Start: Installation & Usage
	5.	Project Structure
	6.	Detailed Usage
	7.	Data Outputs & Visualizations
	8.	Reinforcement Learning Details
	9.	Statistical Analysis
	10.	Future Enhancements & Contributing
	11.	License

Project Overview

This application simulates the game of Blackjack under various settings (number of decks, shuffle point, splits allowed, 
etc.) and offers:
	•	Reinforcement Learning models (Q-Learning, Sarsa), both pretrained and from-scratch training.
	•	Classic Approaches: Basic Strategy (deterministic policy) and a Random agent.
	•	Betting Styles: Flat betting or a spread keyed to the (true) count of the shoe.
	•	Analytics & Statistics:
	•	Plots for bankroll trajectory and EV (average profit/hand).
	•	Options for multi-method comparisons and ANOVA-based tests to see significant differences across strategies.

The front end is built with Flask, presenting a clean UI for parameter selection and generating dynamic plots. Data logs are stored 
during runs and can be exported for deeper analysis.

Key Features
	1.	Flexible Game Rules
	•	Choose 1-4 decks, set shuffle point (e.g., 50% or 75%), define maximum splits, etc.
	2.	Multiple Player Agents
	•	QLearning (fresh or pretrained), Sarsa (fresh or pretrained), Basic Strategy, or Random.
	3.	Betting Options
	•	Flat bets or Count-based spread (customizable for each true count range).
	•	Bankroll clamping for min=1 and max=100,000 (adjustable).
	4.	Interactive Web Interface
	•	Tweak parameters, run single simulations, or compare 4 methods automatically.
	•	Track stats like final bankroll, win/loss/push rates, variance, etc.
	5.	Data Logging & Visualization
	•	Real-time plots of Bankroll vs. Hand Number and EV vs. Hand Number.
	•	Compare multiple methods on shared plots.
	•	Optionally see Epsilon convergence (if QLearning/Sarsa are used).
	•	Download CSV or ZIP containing logs and final summaries.

Technical Stack & Requirements
	•	Python 3.7+ (tested up to Python 3.10)
	•	Flask (for the web server and routes)
	•	matplotlib and scipy/stats (for plotting and statistical analysis)
	•	RQ (Redis Queue) optional for background tasks like repeated simulations/ANOVA
	•	Redis if using the background ANOVA tasks

Python dependencies:

Flask==2.2.2
matplotlib==3.6.0
numpy==1.23.4
pandas==1.5.0
scipy==1.9.3
redis==4.3.4
rq==1.10.1


Quick Start: Installation & Usage
	1.	Clone the Repository

git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName


	2.	(Optional) Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows


	3.	Install Dependencies

pip install -r requirements.txt


	4.	Run the Flask App

python app.py

	•	By default, Flask is served at http://127.0.0.1:5000/.

	5.	Navigate to http://127.0.0.1:5000/ to view the homepage.
	•	Go to “RL Simulation” to start configuring runs.

Project Structure

A brief overview of key files/folders:

.
├── app.py                       # Main Flask app & routes (single-run, compare, etc.)
├── analytics.py                 # DataLogger & plotting (bankroll, EV, epsilon, etc.)
├── rl_agent.py                  # RL agent classes: QLearning, Sarsa, BasicStrategy, Random
├── strategy.py                  # Wagering logic (flat/spread), bankroll updates, etc.
├── environment/                 # Deck creation, card dealing, counting, etc.
├── templates/                   # HTML templates (Flask) 
├── static/                      # CSS, JS, and generated images (plots)
├── tasks.py                     # RQ tasks for background computations (ANOVA, repeated runs)
├── user_agents/                 # Directory for saving/loading pickled RL agents
├── config.py                    # Global configuration (e.g., STARTING_BANKROLL, RL_METHOD)
├── main.py                      # run_main_simulation logic orchestrating environment + strategy
└── README.md                    # This file

Detailed Usage
	1.	Single Run
	•	Select game rules: number of decks, shuffle point, max splits, etc.
	•	Choose an agent (e.g., QLearning, BasicStrategy, etc.).
	•	If QLearning/Sarsa is fresh, set RL hyperparameters (alpha, gamma, epsilon, decay).
	•	Choose flat or spread betting, plus initial bankroll.
	•	Click “Run Simulation.”
	•	Plots (bankroll_history.png and ev_over_time.png) are updated, and a summary table appears.
	2.	Compare All Methods
	•	Same parameter inputs but click “Compare Player Agents.”
	•	The system automatically runs 4 internal simulations (BasicStrategy, Random, QLearning, Sarsa) using the 
same config, then plots combined results.
	3.	Advanced Features
	•	ANOVA:  Repeated simulations for each method are executed via a background job, culminating in an ANOVA test with post-hoc pairwise t-tests.
	•	Reports:  CSV or ZIP exports with summary metrics, individual hand logs, and shoe-level data.

Data Outputs & Visualizations
	•	Bankroll vs. Hand Number (static/bankroll_history.png):
A line chart showing how the bankroll changes over successive hands in a single run.
	•	EV (Average Profit/Hand) vs. Hand Number (static/ev_over_time.png):
Cumulative average profit per hand over the run.
	•	Epsilon Convergence (static/epsilon_convergence.png):
If QLearning or Sarsa is used, tracks how the exploration parameter (epsilon) evolves by hand.
	•	Comparison Plots (static/compare_bankroll.png, static/ev_compare.png):
Multi-line graphs plotting bankroll or EV for each method side by side.

Reinforcement Learning Details
	1.	QLearningAgent
	•	By default, actions are 'hit' and 'stand'.
	•	Q-Table keyed by (player_value, dealer_upcard, is_soft, true_count_int, action) → Q-value.
	•	Customizable alpha, gamma, epsilon, and epsilon_decay.
	2.	SarsaAgent
	•	Similar state representation but follows the SARSA update formula.
	•	Also uses alpha/gamma/epsilon.
	•	For pretrained agents, epsilon = 0.0 to act greedily.
	3.	BasicStrategyAgent
	•	Uses a hand-coded chart (or logic) akin to standard Basic Strategy.
	4.	RandomAgent
	•	Uniformly picks 'hit' or 'stand' from available moves (and might do 'double' or 'split' if allowed).

Statistical Analysis
	•	ANOVA (One-way analysis of variance):
	•	Compares the EV per hand distributions across multiple runs of each method.
	•	If p < 0.05 and there are more than 2 methods, post-hoc t-tests are performed (Bonferroni corrected).
	•	Confidence Intervals:
	•	The system can compute 95% CI around EV for each method.

Future Enhancements & Contributing
	•	To-Do
	•	Expand upon RL with advanced algorithms (e.g., Deep Q-Networks).
	•	Add card animation for a more interactive front end.
	•	Extend the environment to incorporate side bets or multiple players.
	•	Contributions
	•	This project began as a data science capstone.
	•	If interested, fork the repo, open an issue or pull request with suggestions/fixes.

License

Unless otherwise specified, this project is available under the MIT License.
You are free to use, modify, and distribute this code, so long as you include attribution and the license text.

Contact / Further Info
	•	Author: Donald Bango
	•	Email: donaldbango11@aol.com
	•	GitHub: dwbango

Please feel free to open an Issue on this repository if you have questions or encounter any problems.

Thank you for using & exploring this Blackjack RL & Statistical Analysis project!
