# app.py

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

# Disable caching for all responses
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'password')

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
    Welcome page (public).
    """
    return render_template('welcome.html')


@app.route('/simulation')
@login_required
def simulation_page():
    """
    Renders the main simulation page with forms to set parameters.
    """
    return render_template('simulation.html')


@app.route('/run_simulation', methods=['POST'])
@login_required
def run_simulation():
    """
    Handles running a single simulation with chosen parameters (one method).
    """
    rl_method = request.form.get('rl_method', 'BasicStrategy')
    num_decks = request.form.get('num_decks', '2')
    max_splits = request.form.get('max_splits', '3')
    betting_style = request.form.get('betting_style', 'flat')
    num_shoes_str = request.form.get('num_shoes', '100')
    shuffle_point_str = request.form.get('shuffle_point', '0.25')

    print("DEBUG: Form Submission Received:")
    print("  RL Method:", rl_method)
    print("  Num Decks:", num_decks)
    print("  Max Splits:", max_splits)
    print("  Betting Style:", betting_style)
    print("  Num Shoes:", num_shoes_str)
    print("  Shuffle Point:", shuffle_point_str)

    # Update config
    config.RL_METHOD = rl_method
    config.NUM_DECKS = int(num_decks)
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = int(max_splits)
    config.BETTING_STYLE = betting_style
    config.NUM_SHOES_TO_PLAY = int(num_shoes_str)
    config.SHUFFLE_POINT = float(shuffle_point_str)

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

    # Run a single simulation
    results, agent, logger = run_main_simulation()

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
    Runs multiple methods (BasicStrategy, Random, QLearning, Sarsa) under the same config.
    Returns JSON with final summary stats for each method.
    """
    num_decks = int(request.form.get('num_decks', '2'))
    max_splits = int(request.form.get('max_splits', '3'))
    betting_style = request.form.get('betting_style', 'flat')
    num_shoes = int(request.form.get('num_shoes', '100'))
    shuffle_pt = float(request.form.get('shuffle_point', '0.25'))

    # Common config
    config.NUM_DECKS = num_decks
    config.TOTAL_CARDS = 52 * num_decks
    config.MAX_SPLITS = max_splits
    config.BETTING_STYLE = betting_style
    config.NUM_SHOES_TO_PLAY = num_shoes
    config.SHUFFLE_POINT = shuffle_pt

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

    results_dict = {}

    # 1) BasicStrategy
    config.RL_METHOD = "BasicStrategy"
    bs_results, _, _ = run_main_simulation()
    results_dict["BasicStrategy"] = bs_results["summary"]

    # 2) Random
    config.RL_METHOD = "Random"
    rand_results, _, _ = run_main_simulation()
    results_dict["Random"] = rand_results["summary"]

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
        ql_results, _, _ = run_main_simulation(agent_override=qlearning_agent)
        results_dict["QLearning"] = ql_results["summary"]
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
        sarsa_results, _, _ = run_main_simulation(agent_override=sarsa_agent)
        results_dict["Sarsa"] = sarsa_results["summary"]
    else:
        results_dict["Sarsa"] = {"error": "No trained_sarsa.pkl found"}

    return jsonify(results_dict)


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
    Single CSV with summary at top, then hand-level data, then shoe-level data.
    """
    global current_results, current_logger
    if current_results is None or current_logger is None:
        return "No results or logger available. Please run the simulation first."

    summary = current_results.get('summary', {})
    records = current_logger.get_data()
    shoe_records = current_logger.shoe_records

    output = io.StringIO()
    output.write("### Summary Metrics ###\n")
    output.write("Metric,Value\n")
    for key, value in summary.items():
        output.write(f"{key},{value}\n")

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
        dpb = str(r.get("did_player_bust",""))
        ddb = str(r.get("did_dealer_bust",""))
        fps = str(r.get("final_player_hand_size",""))
        fds = str(r.get("final_dealer_hand_size",""))
        is_soft_str = str(r.get("initial_soft_hand",""))
        pc1 = r.get("player_card_1","")
        pc2 = r.get("player_card_2","")
        dealer_up_str = r.get("dealer_upcard","")
        isp = str(r.get("is_pair",""))
        fpc = r.get("final_player_cards","")
        fdc = r.get("final_dealer_cards","")

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
            is_soft_str,
            pc1,
            pc2,
            dealer_up_str,
            isp,
            dpb,
            ddb,
            fps,
            fds,
            fpc,
            fdc
        ]
        output.write(",".join(row) + "\n")

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
    Return a .zip with separate CSVs (summary, hands, shoes).
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
        "dealer_final_total","player_final_total","dealer_blackjack","player_blackjack",
        "shoe_number","original_bet","did_split","did_double",
        "initial_soft_hand","player_card_1","player_card_2","dealer_upcard",
        "is_pair","did_player_bust","did_dealer_bust",
        "final_player_hand_size","final_dealer_hand_size",
        "final_player_cards","final_dealer_cards"
    ]
    hands_io.write(",".join(hand_headers) + "\n")

    for r in records:
        dpb = str(r.get("did_player_bust",""))
        ddb = str(r.get("did_dealer_bust",""))
        fps = str(r.get("final_player_hand_size",""))
        fds = str(r.get("final_dealer_hand_size",""))
        is_soft_str = str(r.get("initial_soft_hand",""))
        pc1 = r.get("player_card_1","")
        pc2 = r.get("player_card_2","")
        dealer_up_str = r.get("dealer_upcard","")
        isp = str(r.get("is_pair",""))
        fpc = r.get("final_player_cards","")
        fdc = r.get("final_dealer_cards","")

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
            is_soft_str,
            pc1,
            pc2,
            dealer_up_str,
            isp,
            dpb,
            ddb,
            fps,
            fds,
            fpc,
            fdc
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

    # Build the ZIP
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
    # Only valid for RL methods
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Strategy charts not available for BasicStrategy or Random."

    global current_agent
    if current_agent is None:
        return ("No RL agent loaded. Please run the simulation first so we have a trained agent to generate strategy charts.")

    analytics.generate_all_strategy_charts(current_agent)
    return "All 3 strategy charts generated!"


if __name__ == '__main__':
    app.run(debug=True)