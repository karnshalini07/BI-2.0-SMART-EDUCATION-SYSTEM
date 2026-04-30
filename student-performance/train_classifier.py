import csv
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_synthetic.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def load_csv_data(csv_path):
    features = []
    labels = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features.append(
                [
                    float(row["attendance"]),
                    float(row["study_hours"]),
                    float(row["assignments_completed"]),
                ]
            )
            labels.append(row["performance"])

    return features, labels


def main():
    X, y_text = load_csv_data(DATA_PATH)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=60,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    artifact = {"model": model, "label_encoder": label_encoder}
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
