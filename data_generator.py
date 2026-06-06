"""
data_generator.py
Generates realistic synthetic monthly climate data from 1950–2023.
Simulates real ENSO cycles with major events at 1972, 1982, 1997, 2015.
"""

import numpy as np
import pandas as pd


def generate_climate_data() -> pd.DataFrame:
    np.random.seed(42)

    # Date range: Jan 1950 – Dec 2023 = 888 months
    dates = pd.date_range(start="1950-01-01", end="2023-12-01", freq="MS")
    n = len(dates)

    # --- Build a realistic ONI signal ---
    t = np.arange(n)

    # Base ENSO cycle: mix of 3.5-yr and 5-yr periodicities
    cycle1 = 0.6 * np.sin(2 * np.pi * t / 42)   # ~3.5-year
    cycle2 = 0.4 * np.sin(2 * np.pi * t / 60)   # ~5-year
    cycle3 = 0.25 * np.sin(2 * np.pi * t / 84)  # ~7-year

    oni_base = cycle1 + cycle2 + cycle3

    # Spike in major El Niño years
    major_events = {
        1972: 1.8,
        1982: 2.2,
        1987: 1.5,
        1991: 1.2,
        1997: 2.8,
        2002: 1.3,
        2009: 1.0,
        2015: 2.6,
        2018: 0.8,
    }
    oni_spikes = np.zeros(n)
    for year, amplitude in major_events.items():
        center = (year - 1950) * 12 + 9  # peak around Oct–Nov
        width = 8
        for i in range(max(0, center - 20), min(n, center + 20)):
            oni_spikes[i] += amplitude * np.exp(-0.5 * ((i - center) / width) ** 2)

    oni_raw = oni_base + oni_spikes
    # Add noise
    oni = oni_raw + np.random.normal(0, 0.15, n)

    # --- Derive correlated climate variables ---
    # SST anomaly closely follows ONI
    sst_anomaly = oni * 0.95 + np.random.normal(0, 0.12, n)

    # SOI is anti-correlated with ONI (negative during El Niño)
    soi = -oni * 8.0 + np.random.normal(0, 2.5, n)

    # Ocean Heat Content correlates with ONI, slightly lagged
    lag = np.roll(oni, 2)
    ohc = lag * 1.8 + np.random.normal(0, 0.4, n)

    # Pressure anomaly also anti-correlates with ONI
    pressure_anomaly = -oni * 1.2 + np.random.normal(0, 0.5, n)

    # Wind speed anomaly
    wind_speed = 6.0 - oni * 0.8 + np.random.normal(0, 0.6, n)
    wind_speed = np.clip(wind_speed, 2.0, 14.0)

    # Humidity slightly elevated during El Niño in some regions
    humidity = 72.0 + oni * 2.5 + np.random.normal(0, 3.0, n)
    humidity = np.clip(humidity, 50.0, 95.0)

    # Rainfall: increases in central Pacific, decreases elsewhere
    rainfall = 180.0 + oni * 15.0 + np.random.normal(0, 25.0, n)
    rainfall = np.clip(rainfall, 20.0, 500.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "SST_anomaly": np.round(sst_anomaly, 3),
            "SOI": np.round(soi, 2),
            "OHC": np.round(ohc, 3),
            "pressure_anomaly": np.round(pressure_anomaly, 3),
            "wind_speed": np.round(wind_speed, 2),
            "humidity": np.round(humidity, 1),
            "rainfall": np.round(rainfall, 1),
            "ONI": np.round(oni, 3),
        }
    )

    return df


if __name__ == "__main__":
    df = generate_climate_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"\nONI stats:\n{df['ONI'].describe()}")
