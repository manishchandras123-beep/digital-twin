import numpy as np

# normal adult vitals - shared starting point for every disease profile
BASELINE = {
    "heart_rate_bpm": 75.0,
    "temperature_C": 37.0,
    "resp_rate_bpm": 16.0,
    "systolic_bp_mmHg": 118.0,
    "spo2_percent": 98.0,
}

NOISE_STD = {
    "heart_rate_bpm": 2.0,
    "temperature_C": 0.1,
    "resp_rate_bpm": 0.8,
    "systolic_bp_mmHg": 3.0,
    "spo2_percent": 0.3,
}

# each profile is how much each vital shifts from baseline at severity=1 (full-blown)
# these are simplified, illustrative patterns - not clinical ground truth
DISEASE_PROFILES = {
    "sepsis": {
        "label": "Sepsis / infection progression",
        "delta": {"heart_rate_bpm": 55, "temperature_C": 3.2, "resp_rate_bpm": 14,
                   "systolic_bp_mmHg": -35, "spo2_percent": -12},
    },
    "cardiac_stress": {
        "label": "Cardiac stress / arrhythmia risk",
        "delta": {"heart_rate_bpm": 65, "temperature_C": 0.3, "resp_rate_bpm": 10,
                   "systolic_bp_mmHg": -45, "spo2_percent": -8},
    },
    "respiratory_distress": {
        "label": "Respiratory distress (e.g. pneumonia)",
        "delta": {"heart_rate_bpm": 35, "temperature_C": 1.8, "resp_rate_bpm": 22,
                   "systolic_bp_mmHg": -15, "spo2_percent": -18},
    },
    "fever_flu": {
        "label": "Fever / flu-like illness",
        "delta": {"heart_rate_bpm": 25, "temperature_C": 3.5, "resp_rate_bpm": 8,
                   "systolic_bp_mmHg": -10, "spo2_percent": -4},
    },
}


def severity_factor(t_fraction):
    # 0 = healthy, 1 = severe. accelerates near the end rather than linear
    if t_fraction <= 0:
        return 0.0
    return float(np.clip(np.exp(4 * (t_fraction - 1)), 0, 1))


def simulate_vitals(disease, severity, rng):
    profile = DISEASE_PROFILES[disease]["delta"]

    heart_rate = BASELINE["heart_rate_bpm"] + severity * profile["heart_rate_bpm"] + rng.normal(0, NOISE_STD["heart_rate_bpm"])
    temperature = BASELINE["temperature_C"] + severity * profile["temperature_C"] + rng.normal(0, NOISE_STD["temperature_C"] * (1 + severity))
    resp_rate = BASELINE["resp_rate_bpm"] + severity * profile["resp_rate_bpm"] + rng.normal(0, NOISE_STD["resp_rate_bpm"])
    systolic_bp = BASELINE["systolic_bp_mmHg"] + severity * profile["systolic_bp_mmHg"] + rng.normal(0, NOISE_STD["systolic_bp_mmHg"])
    spo2 = BASELINE["spo2_percent"] + severity * profile["spo2_percent"] + rng.normal(0, NOISE_STD["spo2_percent"] * (1 + severity * 2))
    spo2 = min(spo2, 100.0)

    return {
        "heart_rate_bpm": round(float(heart_rate), 1),
        "temperature_C": round(float(temperature), 2),
        "resp_rate_bpm": round(float(resp_rate), 1),
        "systolic_bp_mmHg": round(float(systolic_bp), 1),
        "spo2_percent": round(float(spo2), 1),
    }
