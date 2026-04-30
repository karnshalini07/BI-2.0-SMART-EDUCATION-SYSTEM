from pathlib import Path
import pickle


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def get_float_input(prompt, min_value, max_value):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value < min_value or value > max_value:
            print(f"Value must be between {min_value} and {max_value}.")
            continue

        return value


def main():
    if not MODEL_PATH.exists():
        print("model.pkl not found. Run train_classifier.py first.")
        return

    with open(MODEL_PATH, "rb") as file:
        artifact = pickle.load(file)
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]

    print("Student Performance Prediction (CLI)")
    attendance = get_float_input("Enter attendance (0-100): ", 0, 100)
    study_hours = get_float_input("Enter study hours (0-10): ", 0, 10)
    assignments_completed = get_float_input(
        "Enter assignments completed (0-100): ", 0, 100
    )

    features = [[attendance, study_hours, assignments_completed]]
    encoded_prediction = model.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]

    print(f"Predicted performance: {predicted_label}")


if __name__ == "__main__":
    main()
