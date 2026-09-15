import sqlite3
from contextlib import contextmanager

DB_PATH = "digital_twin.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                disease TEXT,
                timestamp TEXT NOT NULL,
                heart_rate_bpm REAL,
                temperature_C REAL,
                resp_rate_bpm REAL,
                systolic_bp_mmHg REAL,
                spo2_percent REAL,
                severity_level REAL,
                news2_score INTEGER,
                risk_label TEXT
            )
        """)
        for col in ("patient_name TEXT", "disease TEXT"):
            try:
                conn.execute(f"ALTER TABLE vitals ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass


def insert_reading(reading):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO vitals (
                patient_name, disease, timestamp, heart_rate_bpm, temperature_C, resp_rate_bpm,
                systolic_bp_mmHg, spo2_percent, severity_level,
                news2_score, risk_label
            ) VALUES (:patient_name, :disease, :timestamp, :heart_rate_bpm, :temperature_C, :resp_rate_bpm,
                      :systolic_bp_mmHg, :spo2_percent, :severity_level,
                      :news2_score, :risk_label)
        """, reading)


def get_all_readings(patient_name=None, limit=500):
    with get_connection() as conn:
        if patient_name:
            rows = conn.execute("""
                SELECT * FROM vitals WHERE patient_name = ? ORDER BY id DESC LIMIT ?
            """, (patient_name, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM vitals ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
    return list(reversed([dict(r) for r in rows]))


def clear_patient(patient_name):
    with get_connection() as conn:
        conn.execute("DELETE FROM vitals WHERE patient_name = ?", (patient_name,))


def clear_all():
    with get_connection() as conn:
        conn.execute("DELETE FROM vitals")


if __name__ == "__main__":
    init_db()
    print(f"database initialized at {DB_PATH}")