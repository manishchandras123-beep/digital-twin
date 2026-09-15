import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from patient_simulator import generate_historical_batch
from risk_scoring import calculate_news2

FEATURES = ["heart_rate_bpm", "temperature_C", "resp_rate_bpm", "systolic_bp_mmHg", "spo2_percent"]


def train_model(disease="sepsis", num_readings=1500, seed=7):
    # build training data: simulate a bunch of patients with this disease,
    # label each reading with the rule-based NEWS2 risk, then let the model
    # learn to predict that label straight from the raw vitals
    df = generate_historical_batch(num_readings=num_readings, disease=disease, seed=seed)
    df["risk_label"] = df.apply(lambda row: calculate_news2(row)["risk_label"], axis=1)

    X = df[FEATURES]
    y = df["risk_label"]

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed)
    model.fit(X, y)
    return model


def predict_risk(model, vitals):
    X = pd.DataFrame([{k: vitals[k] for k in FEATURES}])
    pred_label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    proba_dict = dict(zip(model.classes_, proba))
    confidence = proba_dict[pred_label]
    return pred_label, confidence, proba_dict


if __name__ == "__main__":
    model = train_model(disease="sepsis")

    healthy = {"heart_rate_bpm": 75, "temperature_C": 37.0, "resp_rate_bpm": 16,
               "systolic_bp_mmHg": 118, "spo2_percent": 98}
    deteriorating = {"heart_rate_bpm": 128, "temperature_C": 39.5, "resp_rate_bpm": 27,
                      "systolic_bp_mmHg": 88, "spo2_percent": 89}

    print("healthy ->", predict_risk(model, healthy))
    print("deteriorating ->", predict_risk(model, deteriorating))
