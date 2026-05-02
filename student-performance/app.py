import sqlite3
from collections import Counter
from pathlib import Path
import pickle
import os

from flask import Flask, render_template, request

from recommendations import generate_recommendations


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DB_PATH = BASE_DIR / "database.db"

app = Flask(__name__)
model_artifact = None


def init_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attendance REAL NOT NULL,
                study_hours REAL NOT NULL,
                assignments REAL NOT NULL,
                prediction TEXT NOT NULL
            )
            """
        )
        connection.commit()


def load_model():
    global model_artifact

    if model_artifact is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("model.pkl not found. Run train_classifier.py first.")
        with open(MODEL_PATH, "rb") as file:
            model_artifact = pickle.load(file)

    return model_artifact["model"], model_artifact["label_encoder"]


def save_prediction(attendance, study_hours, assignments, prediction):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO student_predictions
            (attendance, study_hours, assignments, prediction)
            VALUES (?, ?, ?, ?)
            """,
            (attendance, study_hours, assignments, prediction),
        )
        connection.commit()


def get_all_records():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, attendance, study_hours, assignments, prediction
            FROM student_predictions
            ORDER BY id DESC
            """
        ).fetchall()

    return rows


def get_dashboard_data():
    records = get_all_records()
    prediction_counts = Counter(record["prediction"] for record in records)

    return {
        "records": records,
        "total_students": len(records),
        "prediction_counts": prediction_counts,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        attendance = float(request.form["attendance"])
        study_hours = float(request.form["study_hours"])
        assignments_value = request.form.get("assignments") or request.form.get(
            "assignments_completed"
        )
        assignments_completed = float(assignments_value)

        model, label_encoder = load_model()
        features = [[attendance, study_hours, assignments_completed]]
        encoded_prediction = model.predict(features)[0]
        prediction = label_encoder.inverse_transform([encoded_prediction])[0]
        save_prediction(attendance, study_hours, assignments_completed, prediction)
        recommendations = generate_recommendations(
            attendance, study_hours, assignments_completed
        )

        return render_template(
            "index.html",
            prediction=prediction,
            recommendations=recommendations,
            form_data=request.form,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            form_data=request.form,
        )


@app.route("/records")
def records():
    dashboard_data = get_dashboard_data()
    return render_template("dashboard.html", **dashboard_data)


@app.route("/dashboard")
def dashboard():
    dashboard_data = get_dashboard_data()
    return render_template("dashboard.html", **dashboard_data)


init_db()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )