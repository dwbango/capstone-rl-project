# app.py
"""
Flask application entry point for the Blackjack simulation.
Includes routes for:
- Single-method simulation (/run_simulation)
- Multi-method comparison (/run_comparisons)
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

# Disable caching globally
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
    Main simulation page: user can run single or multi-method comparisons.
    """
    return render_template('simulation.html')

@app.route('/run_simulation', methods=['POST'])
@login_required
def run_simulation():
    """
    Runs ONE simulation with the chosen config + RL method. 
    Updates global references to the resulting agent/logger/results.
    Returns basic JSON info (#hands, #shoes).
    """
    rl_method = request.form.get('rl_method', 'BasicStrategy')
    num_decks_str = request.form.get('num_decks', '2')
    max_splits_str = request.form.get('max_splits', '3')
    betting_style = request.form.get('betting_style', 'flat')
    num_shoes_str = request.form.get('num_shoes', '100')
    shuffle_point_str = request.form.get('shuffle_point', '0.25')

    # Debug logging
    print("DEBUG: Form Submission Received (Single Run):")
    print("  RL Method:", rl_method)
    print("  Num Decks:", num_decks_str)
    print("  Max Splits:", max_splits_str)
    print("  Betting Style:", betting_style)
    print("  Num Shoes:", num_shoes_str)
    print("  Shuffle Point:", shuffle_point_str)

    # Apply config
    config.RL_METHOD = rl_method
    config.NUM_DECKS = int(num_decks_str)
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = int(max_splits_str)
    config.BETTING_STYLE = betting_style
    config.NUM_SHOES_TO_PLAY = int(num_shoes_str)
    config.SHUFFLE_POINT = float(shuffle_point_str)

    # If flat betting, read flat bet amount
    if betting_style == 'flat':
        flat_bet_str = request.form.get('flat_bet_amount', '10')
        config.DEFAULT_WAGER = int(flat_bet_str)
    else:
        # Spread betting => build new spread dictionary
        new_spread = {}
        for tc in range(-3, 7):
            field_name = f"spread_{tc}"
            value_str = request.form.get(field_name, '10')
            new_spread[tc] = int(value_str)
        config.BET_SPREAD_DICT = new_spread

    # Run single simulation
    results, agent, logger = run_main_simulation()

    # Store references globally so we can e.g. download a report later
    global current_agent, current_logger, current_results
    current_agent = agent
    current_logger = logger
    current_results = results

    return jsonify({
        "hands_played": results["hands_played"],
        "shoes_played": results["shoes_played"]
    })

@app.route('/run_comparisons', methods=['POST'])
@login_required
def run_comparisons():
    """
    Runs multiple methods (BasicStrategy, Random, QLearning, Sarsa) with the same config.
    Generates multi-line comparison charts for bankroll + EV. 
    Returns JSON with each method's final summary stats or errors.
    """
    # Gather form inputs
    num_decks = int(request.form.get('num_decks', '2'))
    max_splits = int(request.form.get('max_splits', '3'))
    betting_style = request.form.get('betting_style', 'flat')
    num_shoes = int(request.form.get('num_shoes', '100'))
    shuffle_pt = float(request.form.get('shuffle_point', '0.25'))

    # Apply config for all methods
    config.NUM_DECKS = num_decks
    config.TOTAL_CARDS = 52 * num_decks
    config.MAX_SPLITS = max_splits
    config.BETTING_STYLE = betting_style
    config.NUM_SHOES_TO_PLAY = num_shoes
    config.SHUFFLE_POINT = shuffle_pt

    # If flat, read bet amount; else build spread
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

    # We'll store final summaries in results_dict,
    # and store DataLogger objects in method_loggers for charting
    results_dict = {}
    method_loggers = {}

    # 1) BasicStrategy
    config.RL_METHOD = "BasicStrategy"
    bs_results, bs_agent, bs_logger = run_main_simulation()
    results_dict["BasicStrategy"] = bs_results["summary"]
    method_loggers["BasicStrategy"] = bs_logger

    # 2) Random
    config.RL_METHOD = "Random"
    rand_results, rand_agent, rand_logger = run_main_simulation()
    results_dict["Random"] = rand_results["summary"]
    method_loggers["Random"] = rand_logger

    # 3) QLearning (greedy if trained)
    config.RL_METHOD = "QLearning"
    qlearning_agent = None
    try:
        with open("trained_qlearning.pkl", "rb") as f:
            qlearning_agent = pickle.load(f)
        qlearning_agent.epsilon = 0.0
        qlearning_agent.epsilon_decay = 1.0
    except FileNotFoundError:
        pass

    if qlearning_agent:
        ql_results, ql_agent, ql_logger = run_main_simulation(agent_override=qlearning_agent)
        results_dict["QLearning"] = ql_results["summary"]
        method_loggers["QLearning"] = ql_logger
    else:
        results_dict["QLearning"] = {"error": "No trained_qlearning.pkl found"}

    # 4) Sarsa (greedy if trained)
    config.RL_METHOD = "Sarsa"
    sarsa_agent = None
    try:
        with open("trained_sarsa.pkl", "rb") as f:
            sarsa_agent = pickle.load(f)
        sarsa_agent.epsilon = 0.0
        sarsa_agent.epsilon_decay = 1.0
    except FileNotFoundError:
        pass

    if sarsa_agent:
        sarsa_results, sarsa_agent, sarsa_logger = run_main_simulation(agent_override=sarsa_agent)
        results_dict["Sarsa"] = sarsa_results["summary"]
        method_loggers["Sarsa"] = sarsa_logger
    else:
        results_dict["Sarsa"] = {"error": "No trained_sarsa.pkl found"}

    # Create multi-method bankroll + EV charts
    analytics.plot_compare_bankroll(method_loggers)  # bankroll_compare.png
    analytics.plot_compare_ev(method_loggers)        # ev_compare.png

    return jsonify(results_dict)

@app.route('/freeplay')
def freeplay():
    """
    Placeholder for a future 'free play' route.
    """
    return "<h1>Free Play Mode!</h1><p>Coming soon...</p>"

@app.route('/help')
def help_page():
    """
    Basic help page for instructions or documentation links.
    """
    return render_template('help.html')

@app.route('/login', methods=['GET','POST'])
def login():
    """
    Simple login route. On success: session['logged_in'] = True.
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
    """
    Clears session data and redirects to home.
    """
    session.clear()
    return redirect('/')

@app.route('/generate_report')
@login_required
def generate_report():
    """
    Creates a single CSV that has:
      - summary metrics at top
      - hand-level data
      - shoe-level data
    Then returns it as a downloadable file.
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

    # --- Hand-Level Records ---
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

    # --- Shoe-Level Records ---
    output.write("\n### Shoe-Level Records ###\n")
    output.write("shoe_number,card_order\n")
    for sr in shoe_records:
        card_order_str = "|".join([f"{c[0]}{c[1][0]}" for c in sr["card_order"]])
        output.write(f"{sr['shoe_number']},{card_order_str}\n")

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_report.csv"}
    )

@app.route('/download_all_csvs')
@login_required
def download_all_csvs():
    """
    Provides separate CSVs (summary.csv, hands.csv, shoes.csv) inside a single ZIP.
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

    # Build a single ZIP containing summary.csv, hands.csv, shoes.csv
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
    Creates 'epsilon_convergence.png' from the latest single-run, if RL.
    """
    if config.RL_METHOD not in ["QLearning","Sarsa"]:
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
    Returns JSON for the current single-run summary so we can show it on-screen.
    """
    if current_results:
        response = make_response(jsonify(current_results["summary"]))
    else:
        response = make_response(jsonify({"error": "No results yet. Run simulation first."}))
    # Force no caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/generate_strategy_charts')
@login_required
def generate_strategy_charts():
    """
    If QLearning/Sarsa, generate the Hard/Soft/Pairs strategy heatmaps using 
    the Q-values in the agent's table.
    """
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Strategy charts not available for BasicStrategy or Random."

    global current_agent
    if current_agent is None:
        return "No RL agent loaded. Please run the simulation first so we can generate charts."

    analytics.generate_all_strategy_charts(current_agent)
    return "All 3 strategy charts generated!"

if __name__ == '__main__':
    # Launch with debug=True for development environment
    app.run(debug=True)