import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import json
import time
import traceback
from glob import glob
from datetime import datetime
import anthropic
from dotenv import load_dotenv
from shared_utils import setup_logger, log_and_print, read_csv_as_dicts, write_csv_from_dicts
from prompts import JUDGE_PROMPT_TEMPLATE

load_dotenv()

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else None
JUDGE_MODEL = "claude-sonnet-4-5"
BATCH_SIZE = 1000
API_KEY = os.getenv("ANTHROPIC_API_KEY")

input_basename = os.path.basename(INPUT_FILE).replace('.csv', '') if INPUT_FILE else "unknown"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../../../output/claude_judge/{input_basename}/{RUN_ID}"
LOG_DIR = f"../../../logs/{input_basename}"
LOG_FILE = f"{LOG_DIR}/judge_claude.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/token_analysis", exist_ok=True)


def generate_qa_id_fallback(row_idx):
    """Generate QA ID from row index when missing from CSV."""
    return f"qa_{row_idx:06d}"


def load_existing_judgments(output_dir, logger):
    """Load previously judged QA pair IDs from ALL timestamp directories."""
    judged_ids = set()

    parent_dir = os.path.dirname(output_dir)

    timestamp_dirs = []
    if os.path.exists(parent_dir):
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                timestamp_dirs.append(item_path)

    if output_dir not in timestamp_dirs:
        timestamp_dirs.append(output_dir)

    all_batch_files = []
    for ts_dir in timestamp_dirs:
        batch_files = sorted(glob(os.path.join(ts_dir, "batch_*.csv")))
        all_batch_files.extend(batch_files)

    if not all_batch_files:
        log_and_print(logger, "No existing batch files found in any timestamp directory")
        return judged_ids

    log_and_print(logger, f"Found {len(all_batch_files)} existing batch files across {len(timestamp_dirs)} timestamp directories")

    for batch_file in all_batch_files:
        try:
            batch_data = read_csv_as_dicts(batch_file)
            for row in batch_data:
                qa_id = row.get('QA_Pair_ID', '').strip()
                if qa_id:
                    rating = row.get('Judge_Rating', '').strip()
                    if rating:
                        try:
                            rating_val = float(rating)
                            if 1 <= rating_val <= 4:
                                judged_ids.add(qa_id)
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            log_and_print(logger, f"Warning: Failed to read {batch_file}: {e}")

    log_and_print(logger, f"Loaded {len(judged_ids)} already-judged QA pairs from all runs")
    return judged_ids


def run_batch_judge(client, qa_pairs, logger):
    """Execute batch judging using Claude API, polling until completion."""
    log_and_print(logger, f"  Creating batch job with {len(qa_pairs)} requests...")

    requests = []
    for i, qa in enumerate(qa_pairs):
        reference_answer = qa['reference_answer'].replace('\\', '\\\\')
        model_answer = qa['model_answer'].replace('\\', '\\\\')

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=qa['question'],
            teacher_a=reference_answer,
            model_a=model_answer
        )

        request = {
            'custom_id': f'req_{i}',
            'params': {
                'model': JUDGE_MODEL,
                'max_tokens': 10000,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            }
        }
        requests.append(request)

    log_and_print(logger, f"  Submitting batch job...")

    batch = client.beta.messages.batches.create(requests=requests)

    log_and_print(logger, f"  Batch job created: {batch.id}")
    log_and_print(logger, f"  Status: {batch.processing_status}")
    log_and_print(logger, f"  Waiting for completion...")

    poll_count = 0
    while True:
        time.sleep(30)
        poll_count += 1

        batch = client.beta.messages.batches.retrieve(batch.id)

        if poll_count % 2 == 0:
            log_and_print(logger, f"    Status: {batch.processing_status} (poll {poll_count})")

        if batch.processing_status == "ended":
            break
        elif batch.processing_status in ["canceled", "expired"]:
            log_and_print(logger, f"  ERROR: Batch job {batch.processing_status}")
            return {qa['id']: {'rating': -1, 'reason': f'Batch {batch.processing_status}'}
                    for qa in qa_pairs}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    log_and_print(logger, f"  Batch job finished: {batch.processing_status}")

    log_and_print(logger, f"  Downloading results using SDK...")

    result_map = {}
    total_input_tokens = 0
    total_output_tokens = 0
    result_entries = []

    for result_entry in client.beta.messages.batches.results(batch.id):
        result_entries.append(result_entry)
        try:
            custom_id = result_entry.custom_id
            result = result_entry.result

            if result.type == 'succeeded':
                message = result.message

                if hasattr(message, 'usage'):
                    usage = message.usage
                    input_tokens = usage.input_tokens if hasattr(usage, 'input_tokens') else 0
                    output_tokens = usage.output_tokens if hasattr(usage, 'output_tokens') else 0
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                else:
                    input_tokens = 0
                    output_tokens = 0

                if message.content and len(message.content) > 0:
                    content_block = message.content[0]
                    text = content_block.text if hasattr(content_block, 'text') else str(content_block)

                    if text.startswith('```'):
                        lines = text.split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == '```':
                            lines = lines[:-1]
                        text = '\n'.join(lines).strip()

                    try:
                        import re
                        json_match = re.search(r'\{[^{}]*"rating"[^{}]*"reason"[^{}]*\}', text)
                        if json_match:
                            text = json_match.group(0)

                        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

                        response_json = json.loads(text)
                        result_map[custom_id] = {
                            'rating': response_json.get('rating', -1),
                            'reason': response_json.get('reason', 'Parse error'),
                            'text': text,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens
                        }
                    except Exception as e:
                        result_map[custom_id] = {
                            'rating': -1,
                            'reason': f'Parse error: {e} (text={text if text else "EMPTY"})',
                            'text': text,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens
                        }
                else:
                    result_map[custom_id] = {
                        'rating': -1,
                        'reason': 'Empty response',
                        'text': '',
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
            elif result.type == 'errored':
                error = result.error
                error_msg = error.message if hasattr(error, 'message') else 'Unknown error'
                result_map[custom_id] = {
                    'rating': -1,
                    'reason': f'API error: {error_msg}',
                    'text': '',
                    'input_tokens': 0,
                    'output_tokens': 0
                }
            else:
                result_map[custom_id] = {
                    'rating': -1,
                    'reason': f'Unknown result type: {result.type}',
                    'text': '',
                    'input_tokens': 0,
                    'output_tokens': 0
                }
        except Exception as e:
            log_and_print(logger, f"  Warning: Failed to parse result: {e}")

    results = {}
    for i, qa in enumerate(qa_pairs):
        custom_id = f'req_{i}'
        qa_id = qa['id']
        if custom_id in result_map:
            results[qa_id] = result_map[custom_id]
        else:
            results[qa_id] = {
                'rating': -1,
                'reason': 'Missing response',
                'text': '',
                'input_tokens': 0,
                'output_tokens': 0
            }

    success_count = sum(1 for r in results.values() if r['rating'] != -1)
    log_and_print(logger, f"  Completed: {success_count} success, {len(results)-success_count} errors")

    total_tokens = total_input_tokens + total_output_tokens
    log_and_print(logger, f"  Token Usage - Input: {total_input_tokens:,}, Output: {total_output_tokens:,}, Total: {total_tokens:,}")

    token_usage = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_tokens
    }

    return results, token_usage, result_entries


def save_detailed_token_analysis(batch_num, qa_pairs, result_entries, results, token_usage, output_dir):
    """Save per-request token usage with full prompts and responses for analysis."""
    log_dir = os.path.join(output_dir, "token_analysis")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"batch_{batch_num:04d}_token_details.json")

    detailed_entries = []
    for qa in qa_pairs:
        qa_id = qa['id']
        result = results.get(qa_id, {})

        entry = {
            'qa_id': qa_id,
            'request': {
                'question': qa['question'],
                'model_answer': qa['model_answer'],
                'reference_answer': qa['reference_answer'],
                'full_prompt': JUDGE_PROMPT_TEMPLATE.format(
                    question=qa['question'],
                    teacher_a=qa['reference_answer'],
                    model_a=qa['model_answer']
                )
            },
            'response': {
                'rating': result.get('rating'),
                'reason': result.get('reason'),
                'full_response_text': result.get('text')
            },
            'tokens': {
                'input_tokens': result.get('input_tokens', 0),
                'output_tokens': result.get('output_tokens', 0),
                'total_tokens': result.get('input_tokens', 0) + result.get('output_tokens', 0)
            }
        }

        detailed_entries.append(entry)

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'batch_number': batch_num,
        'num_qa_pairs': len(qa_pairs),
        'token_summary': token_usage,
        'detailed_entries': detailed_entries
    }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)


def write_batch_output(batch_num, qa_pairs, results, output_dir, logger):
    """Write judging results for this batch to CSV file."""
    output_file = os.path.join(output_dir, f"batch_{batch_num:04d}.csv")

    rows = []
    for qa in qa_pairs:
        qa_id = qa['id']
        result = results.get(qa_id, {'rating': -1, 'reason': 'Missing result'})
        rows.append({
            'QA_Pair_ID': qa_id,
            'Question': qa['question'],
            'Model_Answer': qa['model_answer'],
            'Reference_Answer': qa['reference_answer'],
            'Judge_Rating': result['rating'],
            'Judge_Reason': result['reason']
        })

    fieldnames = ['QA_Pair_ID', 'Question', 'Model_Answer', 'Reference_Answer', 'Judge_Rating', 'Judge_Reason']
    write_csv_from_dicts(output_file, rows, fieldnames)
    log_and_print(logger, f"  Written {len(rows)} results to {output_file}")


def save_token_summary(output_dir, total_stats):
    """Save cumulative token usage summary"""
    summary_file = os.path.join(output_dir, "token_usage_summary.json")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_qa_pairs': total_stats['num_pairs'],
        'total_batches': total_stats['num_batches'],
        'token_usage': {
            'input_tokens': total_stats['input_tokens'],
            'output_tokens': total_stats['output_tokens'],
            'total_tokens': total_stats['total_tokens']
        },
        'average_per_qa': {
            'input_tokens': round(total_stats['input_tokens'] / total_stats['num_pairs'], 2) if total_stats['num_pairs'] > 0 else 0,
            'output_tokens': round(total_stats['output_tokens'] / total_stats['num_pairs'], 2) if total_stats['num_pairs'] > 0 else 0,
            'total_tokens': round(total_stats['total_tokens'] / total_stats['num_pairs'], 2) if total_stats['num_pairs'] > 0 else 0
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    if not INPUT_FILE:
        print("Usage: python judge_claude.py <exploded_csv_file>")
        sys.exit(1)

    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        sys.exit(1)

    logger = setup_logger(LOG_FILE)
    log_and_print(logger, "="*80)
    log_and_print(logger, f"Claude Judge")
    log_and_print(logger, f"Input: {INPUT_FILE}")
    log_and_print(logger, f"Judge Model: {JUDGE_MODEL}")
    log_and_print(logger, f"Output Directory: {OUTPUT_DIR}")
    log_and_print(logger, f"Batch Size: {BATCH_SIZE}")
    log_and_print(logger, "="*80)

    log_and_print(logger, "\nLoading input file...")
    data = read_csv_as_dicts(INPUT_FILE)
    log_and_print(logger, f"Loaded {len(data)} rows from CSV")

    log_and_print(logger, "\nChecking for existing judgments...")
    judged_ids = load_existing_judgments(OUTPUT_DIR, logger)

    log_and_print(logger, "\nCollecting QA pairs to judge...")
    qa_pairs = []
    skipped_from_csv = 0

    for row_idx, row in enumerate(data):
        qa_id = row.get('QA_Pair_ID', '').strip()
        if not qa_id:
            qa_id = generate_qa_id_fallback(row_idx)
            log_and_print(logger, f"  Warning: Row {row_idx} missing QA_Pair_ID, using fallback: {qa_id}")

        if qa_id in judged_ids:
            continue

        existing_rating = str(row.get('Claude_Judge_Rating', '')).strip()
        try:
            rating_val = float(existing_rating) if existing_rating else -1
            if rating_val >= 1 and rating_val <= 4:
                skipped_from_csv += 1
                continue
        except (ValueError, TypeError):
            pass

        question = row.get('Question', '').strip()
        model_answer = row.get('Model Answer', '').strip()
        reference_answer = row.get('Reference Answer', '').strip()

        if question and model_answer and reference_answer:
            qa_pairs.append({
                'id': qa_id,
                'question': question,
                'model_answer': model_answer,
                'reference_answer': reference_answer
            })

    log_and_print(logger, f"Found {len(qa_pairs)} QA pairs to judge")
    log_and_print(logger, f"Skipped {len(judged_ids)} already-judged pairs (from batch files)")
    log_and_print(logger, f"Skipped {skipped_from_csv} already-judged pairs (from input CSV)")

    if not qa_pairs:
        log_and_print(logger, "\nNo new pairs to judge - all done!")
        log_and_print(logger, "="*80)
        return

    client = anthropic.Anthropic(api_key=API_KEY)

    num_batches = (len(qa_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    log_and_print(logger, f"\nProcessing {num_batches} batch(es)...")

    existing_batches = glob(os.path.join(OUTPUT_DIR, "batch_*.csv"))
    start_batch_num = len(existing_batches) + 1

    total_stats = {
        'num_pairs': 0,
        'num_batches': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0
    }

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(qa_pairs))
        batch_qa_pairs = qa_pairs[batch_start:batch_end]

        log_and_print(logger, f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        log_and_print(logger, f"Processing QA pairs {batch_start} to {batch_end-1}")

        results, token_usage, result_entries = run_batch_judge(client, batch_qa_pairs, logger)

        total_stats['num_pairs'] += len(batch_qa_pairs)
        total_stats['num_batches'] += 1
        total_stats['input_tokens'] += token_usage['input_tokens']
        total_stats['output_tokens'] += token_usage['output_tokens']
        total_stats['total_tokens'] += token_usage['total_tokens']

        write_batch_output(start_batch_num + batch_idx, batch_qa_pairs, results, OUTPUT_DIR, logger)

        save_detailed_token_analysis(start_batch_num + batch_idx, batch_qa_pairs, result_entries, results, token_usage, OUTPUT_DIR)
        log_and_print(logger, f"  Token analysis saved to {OUTPUT_DIR}/token_analysis/")

    log_and_print(logger, "\n" + "="*80)
    log_and_print(logger, "Judging complete!")
    log_and_print(logger, f"Results written to: {OUTPUT_DIR}")
    log_and_print(logger, "")
    log_and_print(logger, "Stats:")
    log_and_print(logger, f"  Input tokens:  {total_stats['input_tokens']:,}")
    log_and_print(logger, f"  Output tokens: {total_stats['output_tokens']:,}")
    log_and_print(logger, f"  Total tokens:  {total_stats['total_tokens']:,}")

    save_token_summary(OUTPUT_DIR, total_stats)
    log_and_print(logger, f"\nToken summary saved to: {OUTPUT_DIR}/token_usage_summary.json")
    log_and_print(logger, "="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = setup_logger(LOG_FILE)
        log_and_print(logger, "\n" + "="*80)
        log_and_print(logger, "Fatal Error")
        log_and_print(logger, "="*80)
        log_and_print(logger, f"Error: {str(e)}")
        log_and_print(logger, "\nFull traceback:")
        log_and_print(logger, traceback.format_exc())
        log_and_print(logger, "="*80)
        sys.exit(1)
