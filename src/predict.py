"""
src/predict.py
==============
Prediction and Inference module for Digital Distraction Pattern Predictor.

Provides:
- `predict_distraction(...)`: High-level function that runs the trained pipeline on new usage records.
- Input validation and informative error messaging.
- Confidence scores and actionable digital wellbeing recommendations.
- CLI interface for quick testing directly from the VS Code terminal.
"""

import os
import sys
import argparse
import joblib
import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def locate_model_file(model_path="models/model.joblib"):
    """Locate the exported model artifact safely."""
    candidates = [
        model_path,
        os.path.join("..", model_path),
        os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Model file '{model_path}' not found. "
        "Please run 'python src/train_model.py' first to train and save the pipeline."
    )


def validate_inputs(category, screen_time_min, launches, interactions, hour, day_of_week):
    """Validates input types and ranges with clear error messages."""
    valid_categories = ["Social", "Entertainment", "Productivity", "Utilities"]
    if category not in valid_categories:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {valid_categories}")

    if not isinstance(screen_time_min, (int, float)) or screen_time_min < 0:
        raise ValueError("screen_time_min must be a non-negative number.")

    if not isinstance(launches, int) or launches < 0:
        raise ValueError("launches must be a non-negative integer.")

    if not isinstance(interactions, int) or interactions < 0:
        raise ValueError("interactions must be a non-negative integer.")

    if not isinstance(hour, int) or not (0 <= hour <= 23):
        raise ValueError("hour must be an integer between 0 and 23.")

    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if day_of_week not in valid_days:
        raise ValueError(f"Invalid day_of_week '{day_of_week}'. Must be one of: {valid_days}")


def get_recommendation_and_explanation(predicted_level, category, screen_time_min, hour):
    """Generates an intuitive, domain-based behavioral explanation and recommendation."""
    late_night = (hour >= 22 or hour < 5)

    if predicted_level == "High Distraction":
        explanation = (
            f"This session represents high distraction potential due to "
            f"{screen_time_min:.1f} minutes of {category} app engagement"
            + (" during late-night hours (disturbing sleep cycle)." if late_night else ".")
        )
        recommendation = (
            "Consider setting an app timer (e.g. 20-30 min limit), turning on Do Not Disturb, "
            "and avoiding screen time 1 hour before bed to safeguard attention and rest."
        )
        badge_color = "#e74c3c"
    elif predicted_level == "Moderate Distraction":
        explanation = (
            f"This session shows balanced, moderate engagement ({screen_time_min:.1f} min in {category}). "
            "It is relatively controlled but could drift into deeper distraction if unmonitored."
        )
        recommendation = (
            "Keep an eye on continuous session lengths and take short eye breaks (the 20-20-20 rule)."
        )
        badge_color = "#f39c12"
    else:  # Low Distraction
        explanation = (
            f"This session represents low distraction. Either the usage was brief ({screen_time_min:.1f} min) "
            f"or centered on functional {category} tasks."
        )
        recommendation = (
            "Healthy digital habit detected! Maintain this pattern for high focus and deep work."
        )
        badge_color = "#2ecc71"

    return explanation, recommendation, badge_color


def predict_distraction(
    category,
    screen_time_min,
    launches,
    interactions,
    hour,
    day_of_week,
    model_path="models/model.joblib"
):
    """
    Executes inference for a single digital usage record.

    Returns:
        dict: Predicted distraction level, class probabilities, explanation, and recommendation.
    """
    validate_inputs(category, screen_time_min, launches, interactions, hour, day_of_week)
    resolved_model_path = locate_model_file(model_path)
    pipeline = joblib.load(resolved_model_path)

    # Construct input dataframe matching exact training columns
    input_df = pd.DataFrame([{
        "category": category,
        "screen_time_min": float(screen_time_min),
        "launches": int(launches),
        "interactions": int(interactions),
        "hour": int(hour),
        "day_of_week": day_of_week
    }])

    prediction = pipeline.predict(input_df)[0]
    classes = pipeline.classes_
    probabilities = pipeline.predict_proba(input_df)[0]
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}

    explanation, recommendation, badge_color = get_recommendation_and_explanation(
        prediction, category, screen_time_min, hour
    )

    return {
        "prediction": prediction,
        "probabilities": prob_dict,
        "confidence": round(float(prob_dict[prediction]) * 100, 1),
        "explanation": explanation,
        "recommendation": recommendation,
        "badge_color": badge_color,
        "inputs": input_df.to_dict(orient="records")[0]
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Digital Distraction Pattern for a digital usage session.")
    parser.add_argument("--category", type=str, default="Social", choices=["Social", "Entertainment", "Productivity", "Utilities"])
    parser.add_argument("--screen_time", type=float, default=65.0, help="Screen time in minutes")
    parser.add_argument("--launches", type=int, default=3, help="App launch count")
    parser.add_argument("--interactions", type=int, default=7, help="In-app interactions/taps")
    parser.add_argument("--hour", type=int, default=23, help="Hour of day (0-23)")
    parser.add_argument("--day", type=str, default="Friday", choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    args = parser.parse_args()

    print("=" * 70)
    print(" 🎯 DIGITAL DISTRACTION PATTERN PREDICTOR - CLI INFERENCE")
    print("=" * 70)
    print("Evaluating input scenario:")
    print(f"  - App Category   : {args.category}")
    print(f"  - Screen Time    : {args.screen_time} minutes")
    print(f"  - App Launches   : {args.launches}")
    print(f"  - Interactions   : {args.interactions}")
    print(f"  - Hour of Day    : {args.hour}:00")
    print(f"  - Day of Week    : {args.day}")

    try:
        result = predict_distraction(
            category=args.category,
            screen_time_min=args.screen_time,
            launches=args.launches,
            interactions=args.interactions,
            hour=args.hour,
            day_of_week=args.day
        )

        print("\nPrediction Result:")
        print(f"  ⭐ Predicted Pattern: {result['prediction']} (Confidence: {result['confidence']}%)")
        print("\nClass Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"  - {cls:<22}: {prob*100:.1f}%")
        print(f"\nBehavioral Explanation:\n  {result['explanation']}")
        print(f"\nDigital Wellbeing Recommendation:\n  💡 {result['recommendation']}")
        print("=" * 70)
    except Exception as e:
        print(f"❌ Error during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
