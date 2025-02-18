# app.py

"""
Flask application entry point for the Blackjack simulation.
Includes routes for:
- Single-method simulation (/run_simulation)
- Multi-method comparison (/run_comparisons)
- Statistical testing route (/run_comparisons_stats)
- Generating various reports, CSV downloads, RL strategy charts, etc.
Requires user login for most routes.
"""

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, send_file, make_response
import config
from main import run_main_simulation
import analytics
import os
import io
import zipfile
import pickle  # For loading trained RL agents

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files
app.secret_key = 'super_secret_key_for_session'

# Disable caching in all responses
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'password')

# Keep references to the *latest single-run* agent/logger/results
current_agent = None
current_logger = None
current_results = None

def login_required(f):
    """
    Decorator requiring login for certain routes.
    """
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return Response("Unauthorized. Please login.", 401)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
def welcome():
    """Public welcome page."""
    return render_template('welcome.html')

@app.route('/simulation')
@login_required
def simulation_page():
    """
    Main simulation page (UI) for single-run or multi-method comparisons.
    """
    return render_template('simulation.html')

@app.route('/run_simulation', methods=['POST'])
@login_required
def run_simulation():
    """
    Runs ONE simulation with the chosen strategy (6 possible):
      1) PretrainedQLearning  -> loads trained_qlearning.pkl
      2) PretrainedSarsa      -> loads trained_sarsa.pkl
      3) QLearning (fresh)
      4) Sarsa (fresh)
      5) BasicStrategy
      6) Random
    """
    # The front-end sends "rl_method" with one of these 6 values
    rl_method = request.form.get('rl_method', 'BasicStrategy')

    # Gather other form inputs
    num_decks_str     = request.form.get('num_decks', '2')
    max_splits_str    = request.form.get('max_splits', '3')
    betting_style     = request.form.get('betting_style', 'flat')
    num_shoes_str     = request.form.get('num_shoes', '100')
    shuffle_point_str = request.form.get('shuffle_point', '0.25')

    print("DEBUG: Single-run request with player strategy:", rl_method)

    # Decide if we load a pickled agent or set RL_METHOD directly
    agent_override = None

    if rl_method == "PretrainedQLearning":
        config.RL_METHOD = "QLearning"  # internally treat as QLearning
        try:
            with open("trained_qlearning.pkl", "rb") as f:
                loaded_agent = pickle.load(f)
            loaded_agent.epsilon = 0.0
            loaded_agent.epsilon_decay = 1.0
            agent_override = loaded_agent
        except FileNotFoundError:
            return jsonify({"error": "trained_qlearning.pkl not found in server directory."})

    elif rl_method == "PretrainedSarsa":
        config.RL_METHOD = "Sarsa"
        try:
            with open("trained_sarsa.pkl", "rb") as f:
                loaded_agent = pickle.load(f)
            loaded_agent.epsilon = 0.0
            loaded_agent.epsilon_decay = 1.0
            agent_override = loaded_agent
        except FileNotFoundError:
            return jsonify({"error": "trained_sarsa.pkl not found in server directory."})

    elif rl_method == "QLearning":
        config.RL_METHOD = "QLearning"

    elif rl_method == "Sarsa":
        config.RL_METHOD = "Sarsa"

    else:
        # BasicStrategy or Random
        config.RL_METHOD = rl_method

    # Now apply the rest of the config
    config.NUM_DECKS = int(num_decks_str)
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = int(max_splits_str)
    config.BETTING_STYLE = betting_style
    config.NUM_SHOES_TO_PLAY = int(num_shoes_str)
    config.SHUFFLE_POINT = float(shuffle_point_str)

    # handle betting style
    if betting_style == 'flat':
        flat_bet_str = request.form.get('flat_bet_amount', '10')
        config.DEFAULT_WAGER = int(flat_bet_str)
    else:
        new_spread = {}
        for tc in range(-3, 7):
            field_name = f"spread_{tc}"
            value_str = request.form.get(field_name, '10')
            new_spread[tc] = int(value_str)
        config.BET_SPREAD_DICT = new_spread

    # Run main sim with optional agent_override
    results, agent, logger = run_main_simulation(agent_override=agent_override)

    global current_agent, current_logger, current_results
    current_agent  = agent
    current_logger = logger
    current_results = results

    if "error" in results:
        return jsonify({"error": results["error"]})

    return jsonify({
        "hands_played": results["hands_played"],
        "shoes_played": results["shoes_played"]
    })

@app.route('/run_comparisons', methods=['POST'])
@login_required
def run_comparisons():
    """
    Example placeholder if you want to compare 4 or 5 selected methods.
    This is not fully implemented in this snippet.
    """
    return jsonify({"error": "run_comparisons is not fully implemented."})

@app.route('/run_comparisons_stats', methods=['POST'])
@login_required
def run_comparisons_stats():
    """
    Example placeholder for repeated runs + ANOVA.
    """
    return jsonify({"error": "run_comparisons_stats is not fully implemented."})

@app.route('/freeplay')
def freeplay():
    return "<h1>Free Play Mode!</h1><p>Coming soon...</p>"

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd  = request.form.get('password')
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            session['logged_in'] = True
            return redirect('/')
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/generate_report')
@login_required
def generate_report():
    if current_results is None or current_logger is None:
        return "No results or logger available. Please run the simulation first."
    # Not fully implemented
    return jsonify({"error": "generate_report not implemented snippet."})

@app.route('/download_all_csvs')
@login_required
def download_all_csvs():
    if current_results is None or current_logger is None:
        return "No results or logger available. Please run the simulation first."
    # Not fully implemented
    return jsonify({"error": "download_all_csvs not implemented snippet."})

@app.route('/generate_epsilon_chart')
@login_required
def generate_epsilon_chart():
    """
    Generate epsilon chart if QLearning or Sarsa was used (or Pretrained).
    """
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Epsilon chart not available for BasicStrategy or Random."

    global current_logger
    if current_logger is None:
        return "No logger available. Please run the simulation first."

    if current_logger.epsilon_values:
        analytics.plot_epsilon_convergence(current_logger)
        return "Epsilon chart generated!"
    else:
        return "No epsilon data available for current method!"

@app.route('/get_summary')
@login_required
def get_summary():
    if current_results:
        response = make_response(jsonify(current_results["summary"]))
    else:
        response = make_response(jsonify({"error": "No results yet. Run simulation first."}))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/generate_strategy_charts')
@login_required
def generate_strategy_charts():
    """
    Strategy charts if RL method = QLearning or Sarsa (including Pretrained).
    """
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Strategy charts not available for BasicStrategy or Random."

    global current_agent
    if current_agent is None:
        return "No RL agent loaded. Please run the simulation first."

    analytics.generate_all_strategy_charts(current_agent)
    return "All 3 strategy charts generated!"

if __name__ == '__main__':
    app.run(debug=True)