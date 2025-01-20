# app.py

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
import config
from main import main as run_main_simulation
import analytics
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'

ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'password')

current_agent = None
current_logger = None
current_results = None

def login_required(f):
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return Response("Unauthorized. Please login.", 401)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/simulation')
@login_required
def simulation_page():
    return render_template('simulation.html')

@app.route('/run_simulation', methods=['POST'])
@login_required
def run_simulation():
    """
    Handles the form submission for running the simulation.
    After running, it sets current_agent, current_logger, and current_results
    to the newly trained agent's data, so we can generate charts from that same agent.
    """
    rl_method = request.form.get('rl_method', 'BasicStrategy')
    num_decks = request.form.get('num_decks', '2')
    max_splits = request.form.get('max_splits', '3')
    betting_style = request.form.get('betting_style', 'flat')

    print("DEBUG: Form Submission Received:")
    print("  RL Method:", rl_method)
    print("  Num Decks:", num_decks)
    print("  Max Splits:", max_splits)
    print("  Betting Style:", betting_style)

    # Update config with user choices
    config.RL_METHOD = rl_method
    config.NUM_DECKS = int(num_decks)
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = int(max_splits)
    config.BETTING_STYLE = betting_style

    if betting_style == 'flat':
        flat_bet_str = request.form.get('flat_bet_amount', '10')
        flat_bet_amount = int(flat_bet_str)
        config.DEFAULT_WAGER = flat_bet_amount
        print("DEBUG: Flat bet set to:", flat_bet_amount)
    else:
        new_spread = {}
        for tc in range(-3, 7):
            field_name = f"spread_{tc}"
            value_str = request.form.get(field_name, '10')
            new_spread[tc] = int(value_str)
        config.BET_SPREAD_DICT = new_spread
        print("DEBUG: Spread dictionary:", new_spread)

    # Actually run the main simulation with updated config
    results, agent, logger = run_main_simulation()

    # Store them globally so we can generate charts from the same trained agent
    global current_agent, current_logger, current_results
    current_agent = agent
    current_logger = logger
    current_results = results

    return jsonify({
        "hands_played": results["hands_played"],
        "shoes_played": results["shoes_played"]
    })

@app.route('/freeplay')
def freeplay():
    return """
    <h1>Free Play Mode!</h1>
    <p>Coming soon...</p>
    """

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
    Instead of re-running run_main_simulation() if results are None,
    we now require the user to have run the simulation first.
    """
    global current_results, current_logger
    if current_results is None or current_logger is None:
        return ("No results or logger available. Please run the simulation first "
                "so we have a trained agent and data to report on.")

    summary = current_results.get('summary', {})
    records = current_logger.get_data()
    shoe_records = current_logger.shoe_records

    import io
    output = io.StringIO()
    output.write("### Summary Metrics ###\n")
    output.write("Metric,Value\n")
    for key, value in summary.items():
        output.write(f"{key},{value}\n")

    output.write("\n### Hand-Level Records ###\n")
    hand_headers = [
        "hand_number","outcome","profit","bankroll","actions_taken",
        "dealer_actions","starting_true_count","starting_decks_remaining"
    ]
    output.write(",".join(hand_headers)+"\n")

    for r in records:
        row = [
            str(r["hand_number"]),
            r["outcome"],
            str(r["profit"]),
            str(r["bankroll"]),
            "|".join(r["actions_taken"]) if r["actions_taken"] else "",
            "|".join(r["dealer_actions"]) if r["dealer_actions"] else "",
            str(r["starting_true_count"]),
            str(r["starting_decks_remaining"])
        ]
        output.write(",".join(row)+"\n")

    output.write("\n### Shoe-Level Records ###\n")
    output.write("shoe_number,card_order\n")
    for sr in shoe_records:
        card_order_str = "|".join([f"{c[0]}{c[1][0]}" for c in sr["card_order"]])
        output.write(f"{sr['shoe_number']},{card_order_str}\n")

    csv_data = output.getvalue()
    output.close()

    from flask import Response
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_report.csv"}
    )

@app.route('/generate_epsilon_chart')
@login_required
def generate_epsilon_chart():
    """
    No longer re-runs run_main_simulation(). If we don't have an RL logger, we bail out.
    """
    if config.RL_METHOD not in ["QLearning","Sarsa"]:
        return "Epsilon chart not available for BasicStrategy."

    global current_logger
    if current_logger is None:
        return "No logger available. Please run the simulation first so we can plot epsilon."

    if current_logger.epsilon_values:
        analytics.plot_epsilon_convergence(current_logger)
        return "Epsilon chart generated!"
    else:
        return "No epsilon data available for current method!"

@app.route('/get_summary')
@login_required
def get_summary():
    """
    Provide summary from the last run. Must have done run_simulation first.
    """
    global current_results
    if current_results:
        return jsonify(current_results["summary"])
    else:
        return jsonify({"error": "No results yet. Run simulation first."})

# Single route for generating all 3 strategy charts (Hard, Soft, Pairs)
@app.route('/generate_strategy_charts')
@login_required
def generate_strategy_charts():
    """
    Only runs if RL_METHOD is QLearning or Sarsa. We do NOT re-run run_main_simulation().
    We require user to have run simulation first.
    """
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Strategy charts not available for BasicStrategy."

    global current_agent
    if current_agent is None:
        return ("No RL agent loaded. Please run the simulation first "
                "so we have a trained agent to generate strategy charts.")

    # Use the already trained agent
    analytics.generate_all_strategy_charts(current_agent)
    return "All 3 strategy charts generated!"

if __name__ == '__main__':
    app.run(debug=True)
