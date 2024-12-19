# app.py

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
import config
from main import main as run_main_simulation
import analytics
import os

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session'

# Credentials (for simplicity)
ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS = os.environ.get('ADMIN_PASS', 'password')

current_agent = None
current_logger = None
current_results = None

def login_required(f):
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return Response("Unauthorized", 401)
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

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

@app.route('/run_simulation', methods=['POST'])
@login_required
def run_simulation():
    rl_method = request.form.get('rl_method', 'BasicStrategy')
    num_decks = request.form.get('num_decks', '1')
    max_splits = request.form.get('max_splits', '3')

    num_decks = int(num_decks)
    max_splits = int(max_splits)

    config.RL_METHOD = rl_method
    config.NUM_DECKS = num_decks
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = max_splits

    results, agent, logger = run_main_simulation()
    global current_agent, current_logger, current_results
    current_agent = agent
    current_logger = logger
    current_results = results
    return jsonify({"hands_played": results["hands_played"], "shoes_played": results["shoes_played"]})

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/generate_report')
@login_required
def generate_report():
    global current_results, current_logger
    if current_results is None or current_logger is None:
        res, ag, lg = run_main_simulation()
        current_results = res
        current_logger = lg
    else:
        res = current_results
        lg = current_logger

    summary = res.get('summary', {})
    records = lg.get_data()
    shoe_records = lg.shoe_records

    import io
    output = io.StringIO()
    # Summary Section
    output.write("### Summary Metrics ###\n")
    output.write("Metric,Value\n")
    for key, value in summary.items():
        output.write(f"{key},{value}\n")

    output.write("\n### Hand-Level Records ###\n")
    hand_headers = ["hand_number","outcome","profit","bankroll","actions_taken","dealer_actions","starting_true_count","starting_decks_remaining"]
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

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_report.csv"}
    )

@app.route('/generate_strategy_chart_plot')
@login_required
def generate_strategy_chart_plot():
    # Only proceed if RL_METHOD is QLearning or Sarsa
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Strategy chart not available for BasicStrategy."
    
    global current_agent
    if current_agent is None:
        r, a, l = run_main_simulation()
        current_agent = a

    analytics.generate_strategy_chart_plot(current_agent)
    return "Strategy chart generated!"

@app.route('/generate_epsilon_chart')
@login_required
def generate_epsilon_chart():
    # Only proceed if RL_METHOD is QLearning or Sarsa
    if config.RL_METHOD not in ["QLearning", "Sarsa"]:
        return "Epsilon chart not available for BasicStrategy."

    if current_logger and current_logger.epsilon_values:
        analytics.plot_epsilon_convergence(current_logger)
        return "Epsilon chart generated!"
    else:
        return "No epsilon data available for current method!"

@app.route('/get_summary')
@login_required
def get_summary():
    if current_results:
        return jsonify(current_results["summary"])
    else:
        return jsonify({"error":"No results yet. Run simulation first."})

if __name__ == '__main__':
    app.run(debug=True)
