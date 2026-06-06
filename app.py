# HOW TO RUN:
# 1. pip install -r requirements.txt
# 2. python app.py
# 3. Open http://localhost:5000 in your browser
# That's it! 🎉

import calendar
import os
from datetime import datetime, timedelta

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from data_generator import generate_climate_data
from model import intensity_label, load_models, predict

app = Flask(__name__)
CORS(app)

# Pre-load models and data on startup
print("🌊 Starting El Niño Watch...")
_models, _scaler = load_models()
_climate_df = generate_climate_data()
print("✅ Ready! Visit http://localhost:5000")


def _current_month_features():
    """Return a plausible 'current' reading from the last row of our synthetic dataset."""
    last = _climate_df.iloc[-1]
    now = datetime.now()
    return {
        "SST_anomaly": float(last["SST_anomaly"]),
        "SOI": float(last["SOI"]),
        "OHC": float(last["OHC"]),
        "pressure_anomaly": float(last["pressure_anomaly"]),
        "wind_speed": float(last["wind_speed"]),
        "humidity": float(last["humidity"]),
        "rainfall": float(last["rainfall"]),
        "month": now.month,
    }


def _variable_status(name, value):
    """Return a kid-friendly status tag for a climate variable."""
    thresholds = {
        "SST_anomaly": (-0.5, 0.5, 1.0),
        "SOI": (-10, -5, 5),
        "OHC": (-1.0, 0.5, 1.5),
        "pressure_anomaly": (-1.0, 0.5, 1.0),
        "wind_speed": (3.0, 8.0, 11.0),
        "humidity": (55.0, 80.0, 90.0),
        "rainfall": (50.0, 200.0, 350.0),
    }
    if name not in thresholds:
        return "Normal 😊", "safe-green"
    low, mid, high = thresholds[name]
    if value > high:
        return "Too High ⚠️", "coral"
    elif value < low:
        return "Too Low ❄️", "ocean-light"
    elif value > mid:
        return "A Bit High 🌤️", "warning-yellow"
    else:
        return "Normal 😊", "safe-green"


@app.route("/")
def index():
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    return send_from_directory(templates_dir, "index.html")


@app.route("/api/history")
def history():
    """Return last 36 months of ONI + all climate variables."""
    df = _climate_df.tail(36).copy()
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": row["date"].strftime("%Y-%m"),
                "label": row["date"].strftime("%b %Y"),
                "oni": round(float(row["ONI"]), 3),
                "oni_label": intensity_label(float(row["ONI"])),
                "SST_anomaly": round(float(row["SST_anomaly"]), 3),
                "SOI": round(float(row["SOI"]), 2),
                "OHC": round(float(row["OHC"]), 3),
                "pressure_anomaly": round(float(row["pressure_anomaly"]), 3),
                "wind_speed": round(float(row["wind_speed"]), 2),
                "humidity": round(float(row["humidity"]), 1),
                "rainfall": round(float(row["rainfall"]), 1),
            }
        )
    current_oni = records[-1]["oni"] if records else 0.0
    return jsonify(
        {
            "history": records,
            "kid_friendly_message": (
                "Here's what the ocean has been doing for the past 3 years! 🌊"
                " Each dot shows how warm (or cool) the ocean was that month."
            ),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_route():
    """Accept current month's climate readings, return 4-month forecast."""
    data = request.get_json(force=True) or {}

    # Fall back to synthetic current values if not provided
    defaults = _current_month_features()
    features = {k: data.get(k, defaults[k]) for k in defaults}

    forecast = predict(features)

    now = datetime.now()
    month_names = []
    for i in range(1, 5):
        future = datetime(now.year, now.month, 1)
        m = (now.month - 1 + i) % 12 + 1
        y = now.year + (now.month - 1 + i) // 12
        month_names.append(calendar.month_name[m] + " " + str(y))

    for i, f in enumerate(forecast):
        f["month_name"] = month_names[i]

    max_oni = max(f["oni"] for f in forecast)
    if max_oni < 0.5:
        msg = "Great news! The ocean looks calm for the next 4 months. 😊 Good time for farmers!"
    elif max_oni < 1.0:
        msg = "A small warming is coming. 🌤️ Nothing scary — just save a little extra water!"
    elif max_oni < 1.5:
        msg = "Heads up! Moderate El Niño is coming. ⚠️ Farmers should plan for less rain."
    elif max_oni < 2.0:
        msg = "Strong El Niño ahead! 🔴 Time to prepare — water supplies may be affected."
    else:
        msg = "Very Strong El Niño coming! 🚨 Please prepare now — this could cause floods or droughts."

    return jsonify({"forecast": forecast, "kid_friendly_message": msg})


@app.route("/api/status")
def status():
    """Return current El Niño status."""
    last = _climate_df.iloc[-1]
    oni = float(last["ONI"])
    label = intensity_label(oni)

    feats = _current_month_features()
    vitals = {}
    display_names = {
        "SST_anomaly": {"label": "Ocean Temperature", "emoji": "🌡️", "unit": "°C above normal"},
        "SOI": {"label": "Wind Balance Score", "emoji": "💨", "unit": "index"},
        "OHC": {"label": "Ocean Energy Level", "emoji": "🌊", "unit": "units"},
        "pressure_anomaly": {"label": "Air Pressure", "emoji": "☁️", "unit": "hPa"},
        "wind_speed": {"label": "Wind Strength", "emoji": "💨", "unit": "m/s"},
        "humidity": {"label": "Air Moisture", "emoji": "💧", "unit": "%"},
        "rainfall": {"label": "Rainfall", "emoji": "🌧️", "unit": "mm"},
    }
    for var, meta in display_names.items():
        val = feats[var]
        status_tag, color = _variable_status(var, val)
        vitals[var] = {
            **meta,
            "value": round(val, 2),
            "status_tag": status_tag,
            "color": color,
        }

    # Build sparklines (last 12 months) for each variable
    last12 = _climate_df.tail(12)
    sparklines = {}
    for var in display_names:
        sparklines[var] = [round(float(v), 2) for v in last12[var].values]

    return jsonify(
        {
            "oni": round(oni, 3),
            "label": label,
            "vitals": vitals,
            "sparklines": sparklines,
            "kid_friendly_message": f"Right now the ocean has an Ocean Heat Score of {oni:.2f}. That means: {label}",
        }
    )


@app.route("/api/explain/<int:month>")
def explain(month):
    """Return plain English impact explanations for a forecast month."""
    feats = _current_month_features()
    forecast = predict(feats)

    if month < 1 or month > 4:
        return jsonify({"error": "Month must be 1–4"}), 400

    f = forecast[month - 1]
    oni = f["oni"]

    impacts = _generate_impacts(oni, month)

    return jsonify(
        {
            "month": month,
            "oni": oni,
            "label": f["label"],
            "impacts": impacts,
            "kid_friendly_message": f"Here's what {f['label']} means for different people in Month {month}:",
        }
    )


def _generate_impacts(oni: float, month: int) -> dict:
    if oni < 0.5:
        return {
            "farmers": "🌾 Great news for farmers! Normal rainfall expected. Plant your crops as usual.",
            "water": "💧 Water reservoirs should stay healthy. No special conservation needed.",
            "families": "🏠 No unusual weather expected. Enjoy normal life!",
            "health": "🏥 Normal health conditions. Stay hydrated and use sunscreen as usual.",
        }
    elif oni < 1.0:
        return {
            "farmers": "🌾 Slightly drier than normal. Water your crops a bit more than usual.",
            "water": "💧 Reservoirs may drop slightly. Start saving water now.",
            "families": "🏠 Keep an emergency water supply at home — just in case!",
            "health": "🏥 Slightly higher heat risk. Drink more water and check on elderly neighbours.",
        }
    elif oni < 1.5:
        return {
            "farmers": "🌾 Reduced rainfall likely. Prioritize drought-resistant crops and efficient irrigation.",
            "water": "💧 Significant reservoir drop expected. Water restrictions may be needed.",
            "families": "🏠 Stock 2–3 weeks of water. Have an emergency plan ready.",
            "health": "🏥 Higher risk of heat illness and mosquito-borne diseases. See a doctor at first sign of illness.",
        }
    elif oni < 2.0:
        return {
            "farmers": "🌾 Serious drought risk! Consider crop insurance and switch to drought-hardy varieties.",
            "water": "💧 Water rationing very likely. Fill containers now while supply is good.",
            "families": "🏠 Prepare emergency kits. Know your evacuation routes.",
            "health": "🏥 High heat and disease risk. Vulnerable people should stay indoors during peak hours.",
        }
    else:
        return {
            "farmers": "🌾 Extreme drought expected. Seek government assistance and consider fallowing fields.",
            "water": "💧 Critical water shortage possible. Follow all official water advisories immediately.",
            "families": "🏠 Follow all emergency orders. Keep 1-month water supply if possible.",
            "health": "🏥 Emergency health risk. Follow all public health advisories immediately.",
        }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
