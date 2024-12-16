# app.py
from flask import Flask, render_template, request, jsonify
import config
from main import main as run_main_simulation

app = Flask(__name__)

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

    # Update config with these parameters
    config.RL_METHOD = rl_method
    config.NUM_DECKS = num_decks
    config.TOTAL_CARDS = 52 * config.NUM_DECKS
    config.MAX_SPLITS = max_splits

    # Run the simulation with updated parameters
    results = run_main_simulation()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
