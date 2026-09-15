import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from disease_engine import BASELINE, DISEASE_PROFILES, severity_factor, simulate_vitals


def generate_historical_batch(num_readings=2000, interval_seconds=60, start_time=None,
                               include_deterioration=True, disease="sepsis", seed=42):
    rng = np.random.default_rng(seed)
    start_time = start_time or (datetime.now() - timedelta(seconds=num_readings * interval_seconds))
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(num_readings)]

    rows = []
    for i in range(num_readings):
        t_fraction = (i / num_readings) if include_deterioration else 0.0
        severity = severity_factor(t_fraction)
        vitals = simulate_vitals(disease, severity, rng)
        vitals["timestamp"] = timestamps[i]
        vitals["severity_level"] = round(severity, 4)
        rows.append(vitals)

    df = pd.DataFrame(rows)
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    return df[cols]


class PatientVitalsStream:
    def __init__(self, disease="sepsis", condition_span_readings=500, seed=None):
        self.disease = disease
        self.rng = np.random.default_rng(seed)
        self.condition_span_readings = condition_span_readings
        self.reading_count = 0

    def next_reading(self):
        t_fraction = min(self.reading_count / self.condition_span_readings, 1.0)
        severity = severity_factor(t_fraction)
        vitals = simulate_vitals(self.disease, severity, self.rng)
        vitals["timestamp"] = datetime.now().isoformat()
        vitals["severity_level"] = round(severity, 4)
        self.reading_count += 1
        return vitals


if __name__ == "__main__":
    for disease in DISEASE_PROFILES:
        df = generate_historical_batch(num_readings=50, disease=disease)
        print(disease, "-> final HR:", df.iloc[-1]["heart_rate_bpm"], "final SpO2:", df.iloc[-1]["spo2_percent"])