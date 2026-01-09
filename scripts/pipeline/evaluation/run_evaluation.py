import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from evaluate import load as load_metric
from shared_utils import setup_logger, log_and_print, read_csv_as_dicts, write_csv_from_dicts

COMPUTE_BERTSCORE = False
COMPUTE_ROUGEL = False
COMPUTE_ENSEMBLE = True
INCREMENTAL_WRITE = True


def get_log_file(csv_path):
    """Get log file path based on input CSV name."""
    model_name = os.path.basename(csv_path).replace('.csv', '')
    log_dir = f"../../../logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    return f"{log_dir}/evaluate.log"



def compute_metrics(data, logger, csv_file):
    """Compute BERTScore and ROUGE-L for all model answers."""
    log_and_print(logger, "\n--- Computing Metrics ---")

    metrics_to_compute = []
    if COMPUTE_BERTSCORE:
        metrics_to_compute.append("BERTScore F1")
    if COMPUTE_ROUGEL:
        metrics_to_compute.append("ROUGEL")

    if not metrics_to_compute:
        log_and_print(logger, "All metrics disabled via toggle variables")
        return list(data[0].keys()) if data else []

    log_and_print(logger, f"Metrics enabled: {', '.join(metrics_to_compute)}")

    fieldnames = list(data[0].keys()) if data else []
    for metric_col in metrics_to_compute:
        if metric_col not in fieldnames:
            fieldnames.append(metric_col)
            for row in data:
                row[metric_col] = ""

    rows_to_compute = []
    for i, row in enumerate(data):
        needs_compute = False
        if COMPUTE_BERTSCORE and not row.get("BERTScore F1", "").strip():
            needs_compute = True
        if COMPUTE_ROUGEL and not row.get("ROUGEL", "").strip():
            needs_compute = True

        if not needs_compute:
            continue

        reference_answer = row.get("Reference Answer", "").strip()
        model_answer = row.get("Model Answer", "").strip()

        if reference_answer and model_answer:
            rows_to_compute.append({
                'data_idx': i,
                'reference_answer': reference_answer,
                'model_answer': model_answer
            })

    log_and_print(logger, f"Found {len(rows_to_compute)} rows needing metrics")

    if not rows_to_compute:
        log_and_print(logger, "All metrics already computed")
        return fieldnames

    log_and_print(logger, "Loading metrics...")
    bertscore = None
    rouge = None
    if COMPUTE_BERTSCORE:
        bertscore = load_metric("bertscore")
    if COMPUTE_ROUGEL:
        rouge = load_metric("rouge")

    BATCH_SIZE = 1000
    for batch_start in range(0, len(rows_to_compute), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(rows_to_compute))
        batch = rows_to_compute[batch_start:batch_end]

        log_and_print(logger, f"  Processing batch {batch_start//BATCH_SIZE + 1}: rows {batch_start+1}-{batch_end}")

        predictions = [row_data['model_answer'] for row_data in batch]
        references = [row_data['reference_answer'] for row_data in batch]

        bert_results = None
        rouge_results = None
        if COMPUTE_BERTSCORE:
            bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
        if COMPUTE_ROUGEL:
            rouge_results = rouge.compute(predictions=predictions, references=references, use_aggregator=False)

        for i, row_data in enumerate(batch):
            data_idx = row_data['data_idx']
            if COMPUTE_BERTSCORE and bert_results:
                data[data_idx]["BERTScore F1"] = str(bert_results['f1'][i])
            if COMPUTE_ROUGEL and rouge_results:
                data[data_idx]["ROUGEL"] = str(rouge_results['rougeL'][i])

        if INCREMENTAL_WRITE:
            log_and_print(logger, f"    Writing batch results to disk...")
            write_csv_from_dicts(csv_file, data, fieldnames)

    log_and_print(logger, "Metrics computation complete")
    return fieldnames



def majority_vote(ratings):
    """Return most common rating across judges, ties go to higher rating."""
    valid_ratings = [r for r in ratings if r not in [-1, '-1', '']]

    if len(valid_ratings) == 0:
        return -1

    valid_ratings = [int(float(r)) for r in valid_ratings]

    mode_result = stats.mode(valid_ratings, keepdims=True)
    return int(mode_result.mode[0])


def add_ensemble_judge(data, logger):
    """Add ensemble rating column using majority vote across all judges."""
    log_and_print(logger, "\n--- Adding Ensemble Judge ---")

    if not COMPUTE_ENSEMBLE:
        log_and_print(logger, "Ensemble judge disabled via toggle variable")
        return list(data[0].keys())

    judge_cols = []
    for judge in ['Claude_Judge_Rating', 'Gemini_Judge_Rating', 'Openai_Judge_Rating', 'Gpt4o_Judge_Rating']:
        if judge in data[0]:
            judge_cols.append(judge)

    log_and_print(logger, f"Found {len(judge_cols)} judge columns: {', '.join(judge_cols)}")

    if len(judge_cols) < 2:
        log_and_print(logger, "Warning: Need at least 2 judges for ensemble voting. Skipping.")
        return list(data[0].keys())

    fieldnames = list(data[0].keys())
    if 'Ensemble_Judge_Rating' not in fieldnames:
        fieldnames.append('Ensemble_Judge_Rating')

    for row in data:
        ratings = [row.get(col, '') for col in judge_cols]
        row['Ensemble_Judge_Rating'] = majority_vote(ratings)

    log_and_print(logger, "Ensemble judge added")
    return fieldnames



def compute_accuracy(ratings):
    """Compute binarized accuracy: 1-2→0, 3-4→1."""
    ratings = [float(r) for r in ratings if r not in [-1, '-1', '']]
    if len(ratings) == 0:
        return 0.0, 0
    binarized = [1 if r >= 3 else 0 for r in ratings]
    return np.mean(binarized), len(ratings)


def get_rating_distribution(ratings):
    """Get distribution of 1-4 ratings and binarized 0-1 ratings."""
    ratings = [float(r) for r in ratings if r not in [-1, '-1', '']]
    if len(ratings) == 0:
        return {}, {}

    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in ratings:
        rating_dist[int(r)] += 1

    binarized = [1 if r >= 3 else 0 for r in ratings]
    binary_dist = {0: binarized.count(0), 1: binarized.count(1)}

    return rating_dist, binary_dist


def compute_benchmark_scores(data, logger):
    """Calculate and print final accuracy scores by QA type and judge."""
    log_and_print(logger, "\n" + "="*80)
    log_and_print(logger, "BENCHMARK SCORES (Binarized: 1-2→0, 3-4→1)")
    log_and_print(logger, "="*80)

    df = pd.DataFrame(data)

    def print_scores(judge_name, ratings, qa_type_label):
        accuracy, n_samples = compute_accuracy(ratings)
        rating_dist, binary_dist = get_rating_distribution(ratings)

        log_and_print(logger, f"  {qa_type_label}:")
        log_and_print(logger, f"    Binarized Accuracy: {accuracy:.1%} (N={n_samples:,})")
        log_and_print(logger, f"    Rating Distribution: 1={rating_dist[1]}, 2={rating_dist[2]}, 3={rating_dist[3]}, 4={rating_dist[4]}")
        log_and_print(logger, f"    Binary Distribution: Incorrect(1-2)={binary_dist[0]}, Correct(3-4)={binary_dist[1]}")

    if 'Ensemble_Judge_Rating' in df.columns:
        log_and_print(logger, "\n" + "="*80)
        log_and_print(logger, "ENSEMBLE JUDGE")
        log_and_print(logger, "="*80)

        if 'teacher' in df['QA Type'].values:
            teacher_df = df[df['QA Type'] == 'teacher']
            ratings = teacher_df['Ensemble_Judge_Rating'].tolist()
            log_and_print(logger, "\nTeacher QA:")
            print_scores("Ensemble", ratings, "TEACHER")

        synthetic_df = df[df['QA Type'].isin(['claude', 'gpt4o'])]
        if len(synthetic_df) > 0:
            ratings = synthetic_df['Ensemble_Judge_Rating'].tolist()
            log_and_print(logger, "\nSynthetic QA:")
            print_scores("Ensemble", ratings, "SYNTHETIC (Claude+GPT4o)")

        log_and_print(logger, "\nBy QA Source:")
        for qa_type in ['teacher', 'gpt4o', 'claude']:
            if qa_type in df['QA Type'].values:
                subset = df[df['QA Type'] == qa_type]
                ratings = subset['Ensemble_Judge_Rating'].tolist()
                print_scores("Ensemble", ratings, qa_type.upper())

    for judge_col in ['Claude_Judge_Rating', 'Gemini_Judge_Rating', 'Openai_Judge_Rating', 'Gpt4o_Judge_Rating']:
        if judge_col in df.columns:
            judge_name = judge_col.replace('_Judge_Rating', '')
            log_and_print(logger, "\n" + "="*80)
            log_and_print(logger, f"{judge_name.upper()} JUDGE")
            log_and_print(logger, "="*80)

            if 'teacher' in df['QA Type'].values:
                teacher_df = df[df['QA Type'] == 'teacher']
                ratings = teacher_df[judge_col].tolist()
                log_and_print(logger, "\nTeacher QA:")
                print_scores(judge_name, ratings, "TEACHER")

            synthetic_df = df[df['QA Type'].isin(['claude', 'gpt4o'])]
            if len(synthetic_df) > 0:
                ratings = synthetic_df[judge_col].tolist()
                log_and_print(logger, "\nSynthetic QA:")
                print_scores(judge_name, ratings, "SYNTHETIC (Claude+GPT4o)")

            log_and_print(logger, "\nBy QA Source:")
            for qa_type in df['QA Type'].unique():
                subset = df[df['QA Type'] == qa_type]
                ratings = subset[judge_col].tolist()
                print_scores(judge_name, ratings, qa_type.upper())

    log_and_print(logger, "\n" + "="*80)



def main():
    parser = argparse.ArgumentParser(description='Unified evaluation pipeline')
    parser.add_argument('csv_file', help='Path to model CSV file (e.g., ../output/claude_sonnet_4.csv)')
    parser.add_argument('--skip-metrics', action='store_true', help='Skip BERTScore/ROUGE computation')
    parser.add_argument('--skip-ensemble', action='store_true', help='Skip ensemble judge computation')
    parser.add_argument('--skip-scores', action='store_true', help='Skip benchmark score computation')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"ERROR: File not found: {args.csv_file}")
        sys.exit(1)

    LOG_FILE = get_log_file(args.csv_file)
    logger = setup_logger(LOG_FILE)
    log_and_print(logger, "="*80)
    log_and_print(logger, "Unified Evaluation Pipeline")
    log_and_print(logger, f"Input: {args.csv_file}")
    log_and_print(logger, "="*80)

    model_name = os.path.basename(args.csv_file).replace('.csv', '')

    log_and_print(logger, f"\nLoading CSV file...")
    data = read_csv_as_dicts(args.csv_file)
    log_and_print(logger, f"Loaded {len(data)} rows")

    fieldnames = list(data[0].keys()) if data else []


    if not args.skip_metrics:
        log_and_print(logger, "\n" + "="*80)
        log_and_print(logger, "STEP 1: Computing Metrics")
        log_and_print(logger, "="*80)
        fieldnames = compute_metrics(data, logger, args.csv_file)
        if not INCREMENTAL_WRITE:
            write_csv_from_dicts(args.csv_file, data, fieldnames)

    if not args.skip_ensemble:
        log_and_print(logger, "\n" + "="*80)
        log_and_print(logger, "STEP 2: Adding Ensemble Judge")
        log_and_print(logger, "="*80)
        fieldnames = add_ensemble_judge(data, logger)
        write_csv_from_dicts(args.csv_file, data, fieldnames)

    if not args.skip_scores:
        log_and_print(logger, "\n" + "="*80)
        log_and_print(logger, "STEP 3: Computing Benchmark Scores")
        log_and_print(logger, "="*80)
        compute_benchmark_scores(data, logger)

    log_and_print(logger, "\n" + "="*80)
    log_and_print(logger, "Evaluation Complete!")
    log_and_print(logger, f"Output: {args.csv_file}")
    log_and_print(logger, "="*80)


if __name__ == "__main__":
    main()
