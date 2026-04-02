from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

# Load models once at startup
lap_model      = joblib.load('models/lap_time_model.pkl')
pos_model      = joblib.load('models/position_model.pkl')
pit_model      = joblib.load('models/pit_strategy_model.pkl')
perf_df        = pd.read_csv('models/driver_performance.csv')

# ── helpers ──────────────────────────────────────────────
def get_driver_info(session_key):
    r = requests.get(f"https://api.openf1.org/v1/drivers?session_key={session_key}")
    return r.json()

def get_sessions():
    r = requests.get("https://api.openf1.org/v1/sessions?year=2024&session_type=Race")
    return r.json()

# ── routes ───────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "F1 backend is running",
        "routes": ["/sessions", "/compare"]
    })
@app.route('/sessions', methods=['GET'])
def sessions():
    data = get_sessions()
    result = [{
        'session_key':        s['session_key'],
        'circuit_short_name': s.get('circuit_short_name',''),
        'country_name':       s.get('country_name',''),
        'date_start':         s.get('date_start',''),
    } for s in data]
    return jsonify(result)


@app.route('/drivers/<int:session_key>', methods=['GET'])
def drivers(session_key):
    data = get_driver_info(session_key)
    seen = set()
    result = []
    for d in data:
        dn = d.get('driver_number')
        if dn in seen:
            continue
        seen.add(dn)
        result.append({
            'driver_number':    dn,
            'full_name':        d.get('full_name',''),
            'team_name':        d.get('team_name',''),
            'headshot_url':     d.get('headshot_url',''),
            'team_colour':      d.get('team_colour',''),
            'name_acronym':     d.get('name_acronym',''),
        })
    return jsonify(result)


@app.route('/compare', methods=['POST'])
def compare():
    body          = request.json
    session_key   = body['session_key']
    driver1       = int(body['driver1'])
    driver2       = int(body['driver2'])

    def predict_for(driver_number):
        # Lap time prediction
        lap_input   = pd.DataFrame([[driver_number, 30, session_key]],
                        columns=['driver_number','lap_number','session_key'])
        lap_time    = round(float(lap_model.predict(lap_input)[0]), 3)

        # Position prediction
        pos_input   = pd.DataFrame([[driver_number, lap_time, session_key]],
                        columns=['driver_number','avg_lap_time','session_key'])
        position    = round(float(pos_model.predict(pos_input)[0]), 1)

        # Pit strategy prediction
        pit_input   = pd.DataFrame([[driver_number, lap_time, session_key]],
                        columns=['driver_number','avg_lap_time','session_key'])
        pit_stops   = int(pit_model.predict(pit_input)[0])

        # Performance score
        score_row   = perf_df[perf_df['driver_number'] == driver_number]
        perf_score  = round(float(score_row['performance_score'].values[0]), 1) \
                      if not score_row.empty else 0.0

        return {
            'lap_time':      lap_time,
            'position':      position,
            'pit_stops':     pit_stops,
            'perf_score':    perf_score,
        }

    return jsonify({
        'driver1': predict_for(driver1),
        'driver2': predict_for(driver2),
    })


if __name__ == '__main__':
    app.run(debug=True)
