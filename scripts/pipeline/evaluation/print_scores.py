import os
import sys
import csv
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv.field_size_limit(500000)

OUTPUT_DIR = "../../../output"

def calculate_scores(filepath):
    """Calculate teacher and synthetic QA scores from ensemble judge ratings."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    teacher_ratings = []
    synthetic_ratings = []

    for row in rows:
        ensemble_rating = row.get('Ensemble_Judge_Rating', '').strip()
        qa_type = row.get('QA Type', '').strip()

        if not ensemble_rating or ensemble_rating == '-1':
            continue

        try:
            rating = float(ensemble_rating)
        except (ValueError, TypeError):
            continue

        if qa_type == 'teacher':
            teacher_ratings.append(rating)
        elif qa_type in ['gpt4o', 'claude']:
            synthetic_ratings.append(rating)

    def compute_binarized_accuracy(ratings):
        """Compute binarized accuracy: 1-2→0, 3-4→1."""
        if len(ratings) == 0:
            return 0.0
        binarized = [1 if r >= 3 else 0 for r in ratings]
        return sum(binarized) / len(binarized)

    teacher_score = compute_binarized_accuracy(teacher_ratings)
    synthetic_score = compute_binarized_accuracy(synthetic_ratings)

    return teacher_score, synthetic_score


def main():
    csv_files = glob(os.path.join(OUTPUT_DIR, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith('template.csv')]

    if not csv_files:
        print("No CSV files found in output directory")
        return

    results = []
    for filepath in csv_files:
        model_name = os.path.basename(filepath).replace('.csv', '')
        try:
            teacher_score, synthetic_score = calculate_scores(filepath)
            results.append((model_name, teacher_score, synthetic_score))
        except Exception as e:
            results.append((model_name, None, None, str(e)))

    results.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)

    print("Model Rankings (Ensemble Judge, Binarized Accuracy)")
    print("Model Name, Teacher QA, Synthetic QA")
    print("=" * 60)

    for result in results:
        if len(result) == 3:
            model_name, teacher_score, synthetic_score = result
            print(f"{model_name}: {teacher_score:.3f}, {synthetic_score:.3f}")
        else:
            model_name, _, _, error = result
            print(f"{model_name}: ERROR - {error}")


if __name__ == "__main__":
    main()
