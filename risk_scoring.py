# simplified version of NEWS2 (early warning score used in hospitals).
# only scores the 4-5 vitals we have - real NEWS2 also factors in
# consciousness level and supplemental oxygen use


def _score_heart_rate(hr):
    if hr <= 40 or hr >= 131:
        return 3
    if 111 <= hr <= 130:
        return 2
    if (41 <= hr <= 50) or (91 <= hr <= 110):
        return 1
    return 0


def _score_temperature(temp):
    if temp <= 35.0:
        return 3
    if temp >= 39.1:
        return 2
    if (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0):
        return 1
    return 0


def _score_resp_rate(rr):
    if rr <= 8 or rr >= 25:
        return 3
    if 21 <= rr <= 24:
        return 2
    if 9 <= rr <= 11:
        return 1
    return 0


def _score_systolic_bp(sbp):
    if sbp <= 90 or sbp >= 220:
        return 3
    if 91 <= sbp <= 100:
        return 2
    if 101 <= sbp <= 110:
        return 1
    return 0


def _score_spo2(spo2):
    if spo2 <= 91:
        return 3
    if 92 <= spo2 <= 93:
        return 2
    if 94 <= spo2 <= 95:
        return 1
    return 0


def calculate_news2(vitals):
    scores = {
        "heart_rate": _score_heart_rate(vitals["heart_rate_bpm"]),
        "temperature": _score_temperature(vitals["temperature_C"]),
        "resp_rate": _score_resp_rate(vitals["resp_rate_bpm"]),
        "systolic_bp": _score_systolic_bp(vitals["systolic_bp_mmHg"]),
        "spo2": _score_spo2(vitals["spo2_percent"]),
    }
    total = sum(scores.values())

    if total >= 7:
        risk_label = "High Risk"
    elif total >= 5:
        risk_label = "Medium Risk"
    elif total >= 1:
        risk_label = "Low Risk"
    else:
        risk_label = "Normal"

    return {
        "news2_score": total,
        "risk_label": risk_label,
        "component_scores": scores,
    }


if __name__ == "__main__":
    healthy = {"heart_rate_bpm": 75, "temperature_C": 37.0, "resp_rate_bpm": 16,
               "systolic_bp_mmHg": 118, "spo2_percent": 98}
    deteriorating = {"heart_rate_bpm": 128, "temperature_C": 39.5, "resp_rate_bpm": 27,
                      "systolic_bp_mmHg": 88, "spo2_percent": 89}

    print("healthy:", calculate_news2(healthy))
    print("deteriorating:", calculate_news2(deteriorating))
