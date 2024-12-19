# app.py
from flask import Flask, render_template, request, jsonify, Response
import config
from main import main as run_main_simulation
import analytics

app = Flask(__name__)

current_agent = None
current_logger = None
current_results = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
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
    records = lg.get_data()  # Detailed hand-level records
    shoe_records = lg.shoe_records  # Shoe-level records

    import io
    output = io.StringIO()
    # Summary Section
    output.write("### Summary Metrics ###\n")
    output.write("Metric,Value\n")
    for key, value in summary.items():
        output.write(f"{key},{value}\n")

    output.write("\n### Hand-Level Records ###\n")
    # Include headers
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
    # Include headers for shoe records
    output.write("shoe_number,card_order\n")
    for sr in shoe_records:
        card_order_str = "|".join([f"{c[0]}{c[1][0]}" for c in sr["card_order"]])  # e.g. "AHearts" shortened to "AH"
        output.write(f"{sr['shoe_number']},{card_order_str}\n")

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_report.csv"}
    )

@app.route('/generate_strategy_chart_plot')
def generate_strategy_chart_plot():
    global current_agent
    if current_agent is None:
        # If no agent yet, run simulation once
        r, a, l = run_main_simulation()
        current_agent = a

    analytics.generate_strategy_chart_plot(current_agent)
    return "Strategy chart generated!"

@app.route('/generate_epsilon_chart')
def generate_epsilon_chart():
    if current_logger and current_logger.epsilon_values:
        analytics.plot_epsilon_convergence(current_logger)
        return "Epsilon chart generated!"
    else:
        return "No epsilon data available for current method!"

@app.route('/get_summary')
def get_summary():
    if current_results:
        return jsonify(current_results["summary"])
    else:
        return jsonify({"error":"No results yet. Run simulation first."})

if __name__ == '__main__':
    app.run(debug=True)

