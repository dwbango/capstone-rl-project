# app.py
"""
Flask application entry point for the Blackjack simulation.
Includes routes for:
- Single-method simulation (/run_simulation)
- Multi-method comparison (/run_comparisons)
- Statistical testing route (/run_comparisons_stats) - now offloaded to RQ
- Generating various reports, CSV downloads, RL strategy charts, etc.
Requires user login for most routes.
"""

from flask import (
    Flask, render_template, request, jsonify, Response,
    session, redirect, url_for, send_file, make_response
)
import config
from main import run_main_simulation
import analytics

import os
import io
import scipy
import zipfile
import pickle  # For loading trained RL agents

import redis
from rq import Queue
from rq.job import Job

# Import your background-task function
from tasks import run_anova_background

# Configure Redis for RQ
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)
rq_queue = Queue('default', connection=conn)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files
app.secret_key = 'super_secret_key_for_session'

# Disable all caching via response headers
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
    Decorator to require login for certain routes.
    """
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return Response("Unauthorized. Please login.", 401)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
def welcome():
    """
    Public welcome page.
    """
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
    Runs ONE simulation with 6 possible player strategies:
      - "PretrainedQLearning" => loads trained_qlearning.pkl
      - "PretrainedSarsa"     => loads trained_sarsa.pkl
      - "QLearning"
      - "Sarsa"
      - "BasicStrategy"
      - "Random"
    Then applies your chosen config parameters and executes `run_main_simulation()`.
    """
    rl_method = request.form.get('rl_method', 'BasicStrategy')
    num_decks_str     = request.form.get('num_decks', '2')
    max_splits_str    = request.form.get('max_splits', '3')
    betting_style     = request.form.get('betting_style', 'flat')
    num_shoes_str     = request.form.get('num_shoes', '100')
    shuffle_point_str = request.form.get('shuffle_point', '0.25')

    print("DEBUG: Single-run request with player strategy:", rl_method)

    # Decide if we load a pickled agent
    agent_override = None
    if rl_method == "PretrainedQLearning":
        config.RL_METHOD = "QLearning"
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

    # Convert user inputs and clamp number of shoes
    num_decks  = int(num_decks_str)
    max_splits = int(max_splits_str)
    shoes_val  = int(num_shoes_str)
    if shoes_val < 1:
        shoes_val = 1
    elif shoes_val > 30000:
        shoes_val = 30000
    shuffle_pt = float(shuffle_point_str)

    # Apply to config
    config.NUM_DECKS       = num_decks
    config.TOTAL_CARDS     = 52 * num_decks
    config.MAX_SPLITS      = max_splits
    config.NUM_SHOES_TO_PLAY = shoes_val
    config.SHUFFLE_POINT   = shuffle_pt
    config.BETTING_STYLE   = betting_style

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

    # Run the simulation
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
    Runs multiple methods (BasicStrategy, Random, QLearning, Sarsa)
    with the same config to compare final stats, produce charts, etc.
    """
    # (UNCHANGED multi-method code)
    num_decks_str    = request.form.get('num_decks', '2')
    max_splits_str   = request.form.get('max_splits', '3')
    betting_style    = request.form.get('betting_style', 'flat')
    num_shoes_str    = request.form.get('num_shoes', '100')
    shuffle_pt_str   = request.form.get('shuffle_point', '0.25')

    num_decks  = int(num_decks_str)
    max_splits = int(max_splits_str)
    shoes_val  = int(num_shoes_str)
    if shoes_val < 1:
        shoes_val = 1
    elif shoes_val > 500:
        shoes_val = 500

    shuffle_pt = float(shuffle_pt_str)

    config.NUM_DECKS         = num_decks
    config.TOTAL_CARDS       = 52 * num_decks
    config.MAX_SPLITS        = max_splits
    config.BETTING_STYLE     = betting_style
    config.NUM_SHOES_TO_PLAY = shoes_val
    config.SHUFFLE_POINT     = shuffle_pt

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

    results_dict   = {}
    method_loggers = {}

    # BasicStrategy
    config.RL_METHOD = "BasicStrategy"
    bs_results, _, bs_logger = run_main_simulation()
    results_dict["BasicStrategy"] = bs_results["summary"]
    method_loggers["BasicStrategy"] = bs_logger

    # Random
    config.RL_METHOD = "Random"
    rand_results, _, rand_logger = run_main_simulation()
    results_dict["Random"] = rand_results["summary"]
    method_loggers["Random"] = rand_logger

    # QLearning
    config.RL_METHOD = "QLearning"
    qlearning_agent = None
    try:
        with open("trained_qlearning.pkl", "rb") as f:
            qlearning_agent = pickle.load(f)
        qlearning_agent.epsilon      = 0.0
        qlearning_agent.epsilon_decay= 1.0
    except FileNotFoundError:
        pass
    if qlearning_agent:
        ql_results, _, ql_logger = run_main_simulation(agent_override=qlearning_agent)
        results_dict["QLearning"] = ql_results["summary"]
        method_loggers["QLearning"] = ql_logger
    else:
        ql_results, _, ql_logger = run_main_simulation()
        results_dict["QLearning"] = ql_results["summary"]
        method_loggers["QLearning"] = ql_logger

    # Sarsa
    config.RL_METHOD = "Sarsa"
    sarsa_agent = None
    try:
        with open("trained_sarsa.pkl", "rb") as f:
            sarsa_agent = pickle.load(f)
        sarsa_agent.epsilon      = 0.0
        sarsa_agent.epsilon_decay= 1.0
    except FileNotFoundError:
        pass
    if sarsa_agent:
        sarsa_results, _, sarsa_logger = run_main_simulation(agent_override=sarsa_agent)
        results_dict["Sarsa"] = sarsa_results["summary"]
        method_loggers["Sarsa"] = sarsa_logger
    else:
        sarsa_results, _, sarsa_logger = run_main_simulation()
        results_dict["Sarsa"] = sarsa_results["summary"]
        method_loggers["Sarsa"] = sarsa_logger

    analytics.plot_compare_bankroll(method_loggers)
    analytics.plot_compare_ev(method_loggers)

    return jsonify(results_dict)

# ----------------------------------------------------------------
#  REPLACED run_comparisons_stats WITH BACKGROUND ENQUEUE LOGIC
# ----------------------------------------------------------------
@app.route('/run_comparisons_stats', methods=['POST'])
@login_required
def run_comparisons_stats():
    """
    Instead of performing repeated runs inline, enqueue a background job.
    We let tasks.py's run_anova_background handle the heavy-lifting.
    """
    repeats = 30
    shoes_per_run = 50

    job = rq_queue.enqueue(run_anova_background, repeats, shoes_per_run)
    return jsonify({"job_id": job.get_id()})

@app.route('/job_status/<job_id>')
@login_required
def job_status(job_id):
    """
    Endpoint to check the status of a background job by job_id.
    Returns JSON with status and result if finished.
    """
    try:
        job = Job.fetch(job_id, connection=conn)
    except:
        return jsonify({"error": "No such job"}), 404

    if job.is_finished:
        return jsonify({"status": "finished", "result": job.result})
    elif job.is_failed:
        return jsonify({"status": "failed"})
    else:
        return jsonify({"status": "in-progress"})

@app.route('/freeplay')
def freeplay():
    return "<h1>Free Play Mode!</h1><p>Coming soon...</p>"

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/login', methods=['GET','POST'])
def login():
    """
    Simple login. If user=ADMIN_USER and pwd=ADMIN_PASS, set session['logged_in'] = True.
    """
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
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
    """
    Generates a single CSV:
      1) summary metrics
      2) hand-level data
      3) shoe-level data
    for the *latest single-run* simulation.
    """
    global current_results, current_logger
    if current_results is None or current_logger is None:
        return "No results or logger available. Please run the simulation first."

    summary = current_results.get('summary', {})
    records = current_logger.get_data()
    shoe_records = current_logger.shoe_records

    output = io.StringIO()
    # --- Summary ---
    output.write("### Summary Metrics ###\n")
    output.write("Metric,Value\n")
    for key, value in summary.items():
        output.write(f"{key},{value}\n")

    # --- Hand-Level ---
    output.write("\n### Hand-Level Records ###\n")
    hand_headers = [
        "hand_number","outcome","profit","bankroll","actions_taken",
        "dealer_actions","starting_true_count","starting_decks_remaining",
        "dealer_final_total","player_final_total",
        "dealer_blackjack","player_blackjack",
        "shoe_number","original_bet","did_split","did_double",
        "initial_soft_hand","player_card_1","player_card_2","dealer_upcard",
        "is_pair","did_player_bust","did_dealer_bust",
        "final_player_hand_size","final_dealer_hand_size",
        "final_player_cards","final_dealer_cards"
    ]
    output.write(",".join(hand_headers) + "\n")

    for r in records:
        row = [
            str(r["hand_number"]),
            r["outcome"],
            str(r["profit"]),
            str(r["bankroll"]),
            "|".join(r["actions_taken"]) if r["actions_taken"] else "",
            "|".join(r["dealer_actions"]) if r["dealer_actions"] else "",
            str(r["starting_true_count"]),
            str(r["starting_decks_remaining"]),
            str(r.get("dealer_final_total","")),
            str(r.get("player_final_total","")),
            str(r.get("dealer_blackjack","")),
            str(r.get("player_blackjack","")),
            str(r.get("shoe_number","")),
            str(r.get("original_bet","")),
            str(r.get("did_split","")),
            str(r.get("did_double","")),
            str(r.get("initial_soft_hand","")),
            str(r.get("player_card_1","")),
            str(r.get("player_card_2","")),
            str(r.get("dealer_upcard","")),
            str(r.get("is_pair","")),
            str(r.get("did_player_bust","")),
            str(r.get("did_dealer_bust","")),
            str(r.get("final_player_hand_size","")),
            str(r.get("final_dealer_hand_size","")),
            str(r.get("final_player_cards","")),
            str(r.get("final_dealer_cards",""))
        ]
        output.write(",".join(row) + "\n")

    # --- Shoe-Level ---
    output.write("\n### Shoe-Level Records ###\n")
    output.write("shoe_number,card_order\n")
    for sr in shoe_records:
        card_order_str = "|".join([f"{c[0]}{c[1][0]}" for c in sr["card_order"]])
        output.write(f"{sr['shoe_number']},{card_order_str}\n")

    csv_data = output.getvalue()
    output.close()

    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name='simulation_report.csv'
    )

@app.route('/download_all_csvs')
@login_required
def download_all_csvs():
    """
    Returns a ZIP containing summary.csv, hands.csv, and shoes.csv
    for the latest single-run simulation.
    """
    global current_results, current_logger
    if current_results is None or current_logger is None:
        return "No results or logger available. Please run the simulation first."

    summary = current_results["summary"]
    records = current_logger.get_data()
    shoe_records = current_logger.shoe_records

    # summary.csv
    summary_io = io.StringIO()
    summary_io.write("Metric,Value\n")
    for key, value in summary.items():
        summary_io.write(f"{key},{value}\n")
    summary_data = summary_io.getvalue()
    summary_io.close()

    # hands.csv
    hands_io = io.StringIO()
    hand_headers = [
        "hand_number","outcome","profit","bankroll","actions_taken",
        "dealer_actions","starting_true_count","starting_decks_remaining",
        "dealer_final_total","player_final_total",
        "dealer_blackjack","player_blackjack",
        "shoe_number","original_bet","did_split","did_double",
        "initial_soft_hand","player_card_1","player_card_2","dealer_upcard",
        "is_pair","did_player_bust","did_dealer_bust",
        "final_player_hand_size","final_dealer_hand_size",
        "final_player_cards","final_dealer_cards"
    ]
    hands_io.write(",".join(hand_headers) + "\n")

    for r in records:
        row = [
            str(r["hand_number"]),
            r["outcome"],
            str(r["profit"]),
            str(r["bankroll"]),
            "|".join(r["actions_taken"]) if r["actions_taken"] else "",
            "|".join(r["dealer_actions"]) if r["dealer_actions"] else "",
            str(r["starting_true_count"]),
            str(r["starting_decks_remaining"]),
            str(r.get("dealer_final_total","")),
            str(r.get("player_final_total","")),
            str(r.get("dealer_blackjack","")),
            str(r.get("player_blackjack","")),
            str(r.get("shoe_number","")),
            str(r.get("original_bet","")),
            str(r.get("did_split","")),
            str(r.get("did_double","")),
            str(r.get("initial_soft_hand","")),
            str(r.get("player_card_1","")),
            str(r.get("player_card_2","")),
            str(r.get("dealer_upcard","")),
            str(r.get("is_pair","")),
            str(r.get("did_player_bust","")),
            str(r.get("did_dealer_bust","")),
            str(r.get("final_player_hand_size","")),
            str(r.get("final_dealer_hand_size","")),
            str(r.get("final_player_cards","")),
            str(r.get("final_dealer_cards",""))
        ]
        hands_io.write(",".join(row) + "\n")
    hands_data = hands_io.getvalue()
    hands_io.close()

    # shoes.csv
    shoes_io = io.StringIO()
    shoes_io.write("shoe_number,card_order\n")
    for sr in shoe_records:
        card_order_str = "|".join([f"{c[0]}{c[1][0]}" for c in sr["card_order"]])
        shoes_io.write(f"{sr['shoe_number']},{card_order_str}\n")
    shoes_data = shoes_io.getvalue()
    shoes_io.close()

    # Build a ZIP archive in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", summary_data)
        zf.writestr("hands.csv", hands_data)
        zf.writestr("shoes.csv", shoes_data)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='all_reports.zip'
    )

@app.route('/generate_epsilon_chart')
@login_required
def generate_epsilon_chart():
    """
    Generate epsilon chart if QLearning or Sarsa or Pretrained Q/S.
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
    """
    Return the summary from the latest single-run simulation in JSON form.
    """
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
    Generate Hard/Soft/Pairs strategy charts for QLearning or Sarsa agents
    (including if they're pretrained).
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