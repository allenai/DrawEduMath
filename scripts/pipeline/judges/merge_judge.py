"""
Simple, robust judge merge script.
Just updates rating and reason columns - nothing fancy.

Usage: python merge_judge_fixed.py <csv_file> <judge_name>
Example: python merge_judge_fixed.py ../../output/gemini_2.5_pro.csv claude
"""

import os
import sys
import csv
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utils import setup_logger, log_and_print

csv.field_size_limit(500000)


def load_judge_results_from_batches(judge_dir):
    """Load all judge results from batch CSV files."""
    results = {}

    batch_files = glob(os.path.join(judge_dir, "**", "batch_*.csv"), recursive=True)

    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qa_id = row.get('QA_Pair_ID', '').strip() or row.get('QA_ID', '').strip()
                rating = row.get('Judge_Rating', '').strip()
                reason = row.get('Judge_Reason', '').strip()

                if qa_id and rating:
                    try:
                        rating_val = float(rating)
                        if 1 <= rating_val <= 4:
                            results[qa_id] = {'rating': rating, 'reason': reason}
                    except (ValueError, TypeError):
                        pass

    return results, len(batch_files)


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_judge_fixed.py <csv_file> <judge_name>")
        print("Example: python merge_judge_fixed.py ../../output/gemini_2.5_pro.csv claude")
        sys.exit(1)

    csv_file = sys.argv[1]
    judge_name = sys.argv[2].lower()

    if judge_name not in ['claude', 'gemini', 'openai']:
        print(f"ERROR: Invalid judge name '{judge_name}'. Must be: claude, gemini, or openai")
        sys.exit(1)

    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)

    model_name = os.path.basename(csv_file).replace('.csv', '')
    log_dir = f"../../logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(f"{log_dir}/merge_{judge_name}_fixed.log")

    log_and_print(logger, "=" * 80)
    log_and_print(logger, f"FIXED Merge Script - {judge_name.title()} Judge")
    log_and_print(logger, f"Input: {csv_file}")
    log_and_print(logger, "=" * 80)

    rating_col = f"{judge_name.title()}_Judge_Rating"
    reason_col = f"{judge_name.title()}_Judge_Reason"

    judge_dir = f"../../output/{judge_name}_judge/{model_name}"
    if not os.path.exists(judge_dir):
        log_and_print(logger, f"ERROR: Judge directory not found: {judge_dir}")
        sys.exit(1)

    log_and_print(logger, "\nLoading judge results from batch files...")
    judge_results, batch_count = load_judge_results_from_batches(judge_dir)
    log_and_print(logger, f"Loaded {len(judge_results)} valid results from {batch_count} batch files")

    if not judge_results:
        log_and_print(logger, "ERROR: No valid judge results found")
        sys.exit(1)

    log_and_print(logger, "\nReading CSV file...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    log_and_print(logger, f"Read {len(rows)} rows")
    log_and_print(logger, f"Original columns: {len(original_fieldnames)}")

    if rating_col not in original_fieldnames:
        original_fieldnames.append(rating_col)
        log_and_print(logger, f"Added column: {rating_col}")
    if reason_col not in original_fieldnames:
        original_fieldnames.append(reason_col)
        log_and_print(logger, f"Added column: {reason_col}")

    log_and_print(logger, "\nMerging judge results...")
    merged = 0
    skipped_valid = 0
    skipped_no_result = 0

    for row in rows:
        qa_id = row.get('QA_Pair_ID', '').strip()

        if qa_id not in judge_results:
            skipped_no_result += 1
            if rating_col not in row:
                row[rating_col] = ''
            if reason_col not in row:
                row[reason_col] = ''
            continue

        existing_rating = str(row.get(rating_col, '')).strip()
        is_valid = False
        if existing_rating:
            try:
                rating_val = float(existing_rating)
                if 1 <= rating_val <= 4:
                    is_valid = True
            except (ValueError, TypeError):
                pass

        if is_valid:
            skipped_valid += 1
        else:
            row[rating_col] = judge_results[qa_id]['rating']
            row[reason_col] = judge_results[qa_id]['reason']
            merged += 1

    log_and_print(logger, f"  Merged: {merged}")
    log_and_print(logger, f"  Skipped (already valid): {skipped_valid}")
    log_and_print(logger, f"  Skipped (no result): {skipped_no_result}")

    log_and_print(logger, f"\nWriting {len(rows)} rows back to {csv_file}...")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=original_fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows)

    file_size = os.path.getsize(csv_file)
    log_and_print(logger, f"Write successful! File size: {file_size:,} bytes")

    with open(csv_file, 'r', encoding='utf-8') as f:
        verify_rows = sum(1 for _ in csv.DictReader(f))

    log_and_print(logger, f"Verification: {verify_rows} rows in file")

    if verify_rows != len(rows):
        log_and_print(logger, f"WARNING: Row count mismatch! Expected {len(rows)}, got {verify_rows}")
    else:
        log_and_print(logger, "Row count verified [OK]")

    log_and_print(logger, "=" * 80)
    log_and_print(logger, "Merge complete!")
    log_and_print(logger, "=" * 80)


if __name__ == "__main__":
    main()
