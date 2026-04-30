def generate_recommendations(attendance, study_hours, assignments_completed):
    suggestions = []

    if attendance < 75:
        suggestions.append(
            "Improve class attendance to stay consistent with lessons."
        )

    if study_hours < 2:
        suggestions.append(
            "Increase daily study time and follow a regular study schedule."
        )

    if assignments_completed < 70:
        suggestions.append(
            "Complete more assignments on time to strengthen understanding."
        )

    if not suggestions:
        suggestions.append(
            "Keep up the good work. Your current academic habits look strong."
        )

    return suggestions


if __name__ == "__main__":
    sample = generate_recommendations(68, 1.5, 60)
    for item in sample:
        print(f"- {item}")
