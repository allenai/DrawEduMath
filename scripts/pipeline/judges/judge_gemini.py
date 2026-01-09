import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import json
import time
import requests
import traceback
from datetime import datetime
from glob import glob
from dotenv import load_dotenv
from shared_utils import setup_logger, log_and_print, read_csv_as_dicts, write_csv_from_dicts
from prompts import JUDGE_PROMPT_TEMPLATE

load_dotenv()

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else None
JUDGE_MODEL = "models/gemini-2.5-pro"
BATCH_SIZE = 1000
API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta"

input_basename = os.path.basename(INPUT_FILE).replace('.csv', '') if INPUT_FILE else "unknown"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"../../../output/gemini_judge/{input_basename}/{RUN_ID}"
LOG_DIR = f"../../../logs/{input_basename}"
LOG_FILE = f"{LOG_DIR}/judge_gemini.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/token_analysis", exist_ok=True)


def generate_qa_id_fallback(row_idx):
    """Generate fallback QA ID from row index when missing from CSV."""
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


def upload_file(api_key, file_path, display_name, logger):
    """Upload file using REST API with resumable upload"""
    file_size = os.path.getsize(file_path)

    start_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": "application/json",
        "Content-Type": "application/json",
    }
    start_body = json.dumps({'file': {'display_name': display_name}})

    try:
        start_response = requests.post(start_url, headers=start_headers, data=start_body)
        start_response.raise_for_status()

        if 'x-goog-upload-url' not in start_response.headers:
            error_json = start_response.json()
            error_message = error_json.get("error", {}).get("message", "Unknown error")
            raise Exception(f"API Error: {error_message}")

        upload_url = start_response.headers['x-goog-upload-url']

        upload_headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        with open(file_path, 'rb') as f:
            upload_response = requests.post(upload_url, headers=upload_headers, data=f)

        upload_response.raise_for_status()
        response_json = upload_response.json()
        file_name = response_json.get('file', {}).get('name')

        if not file_name:
            raise Exception("File name not found in upload response")

        return file_name

    except requests.exceptions.RequestException as e:
        log_and_print(logger, f"   ERROR: File upload failed: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_details = e.response.json()
                log_and_print(logger, f"   Response: {json.dumps(error_details, indent=2)}")
            except:
                log_and_print(logger, f"   Response: {e.response.text}")
        raise


def run_batch_judge(api_key, qa_pairs, logger):
    """Run batch judging job and return results"""
    log_and_print(logger, f"  Creating batch job with {len(qa_pairs)} requests...")

    temp_dir = "temp_batch_files"
    os.makedirs(temp_dir, exist_ok=True)

    jsonl_path = os.path.join(temp_dir, f"temp_batch_{int(time.time())}.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=qa['question'],
                teacher_a=qa['reference_answer'],
                model_a=qa['model_answer']
            )

            request = {
                'key': qa['id'],
                'request': {
                    'contents': [{
                        'parts': [{'text': prompt}]
                    }]
                }
            }
            f.write(json.dumps(request) + '\n')

    log_and_print(logger, f"  Uploading batch file...")

    try:
        uploaded_file_name = upload_file(api_key, jsonl_path, f'judge-{int(time.time())}', logger)
        log_and_print(logger, f"  File uploaded: {uploaded_file_name}")
    except Exception as e:
        log_and_print(logger, f"  ERROR: File upload failed: {e}")
        return {}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    try:
        create_url = f"{BASE_API_URL}/{JUDGE_MODEL}:batchGenerateContent?key={api_key}"
        create_payload = {
            "batch": {
                "display_name": f'judge-{int(time.time())}',
                "input_config": {
                    "file_name": uploaded_file_name
                }
            }
        }
        response = requests.post(create_url, headers={"Content-Type": "application/json"}, json=create_payload)
        response.raise_for_status()
        batch_job = response.json()
        batch_job_name = batch_job['name']
        batch_job_state = batch_job.get('metadata', {}).get('state')
        log_and_print(logger, f"  Batch job created: {batch_job_name}")
        log_and_print(logger, f"  Status: {batch_job_state}")

    except requests.exceptions.RequestException as e:
        log_and_print(logger, f"  ERROR: Batch creation failed: {e}")
        return {}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    log_and_print(logger, "  Waiting for completion...")

    completed_states = {
        'BATCH_STATE_SUCCEEDED',
        'BATCH_STATE_FAILED',
        'BATCH_STATE_CANCELLED',
        'BATCH_STATE_EXPIRED',
    }

    job_url = f"{BASE_API_URL}/{batch_job_name}?key={api_key}"
    poll_count = 0

    while batch_job_state not in completed_states:
        time.sleep(30)
        poll_count += 1
        try:
            response = requests.get(job_url)
            response.raise_for_status()
            batch_job = response.json()
            batch_job_state = batch_job.get('metadata', {}).get('state')
            if poll_count % 2 == 0:
                log_and_print(logger, f"    Status: {batch_job_state} (poll {poll_count})")
        except requests.exceptions.RequestException as e:
            log_and_print(logger, f"  WARNING: Poll failed: {e}")

    log_and_print(logger, f"  Batch job finished: {batch_job_state}")

    try:
        os.remove(jsonl_path)
    except:
        pass

    if batch_job_state != 'BATCH_STATE_SUCCEEDED':
        error_details = batch_job.get('error', 'Unknown error')
        log_and_print(logger, f"  ERROR: Batch job failed: {error_details}")
        return {qa['id']: {'rating': -1, 'reason': f'Batch failed: {batch_job_state}'}
                for qa in qa_pairs}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    result_file_name = batch_job.get('response', {}).get('responsesFile')
    if not result_file_name:
        log_and_print(logger, "  ERROR: No output file found")
        return {qa['id']: {'rating': -1, 'reason': 'No batch results'}
                for qa in qa_pairs}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    log_and_print(logger, f"  Downloading results...")

    try:
        download_url = f"https://generativelanguage.googleapis.com/download/v1beta/{result_file_name}:download?alt=media&key={api_key}"
        response = requests.get(download_url)
        response.raise_for_status()
        result_content = response.content
    except requests.exceptions.RequestException as e:
        log_and_print(logger, f"  ERROR: Download failed: {e}")
        return {qa['id']: {'rating': -1, 'reason': 'Download failed'}
                for qa in qa_pairs}, {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, []

    result_map = {}
    total_input_tokens = 0
    total_output_tokens = 0
    result_lines = result_content.decode('utf-8').strip().split('\n')

    for line in result_lines:
        if not line.strip():
            continue
        try:
            result_obj = json.loads(line)
            qa_id = result_obj.get('key', '')

            if 'response' in result_obj:
                response = result_obj['response']

                if 'usageMetadata' in response:
                    usage = response['usageMetadata']
                    total_input_tokens += usage.get('promptTokenCount', 0)
                    total_output_tokens += usage.get('candidatesTokenCount', 0)

                if 'candidates' in response and len(response['candidates']) > 0:
                    candidate = response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            text = parts[0]['text'].strip()

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
                                result_map[qa_id] = {
                                    'rating': response_json.get('rating', -1),
                                    'reason': response_json.get('reason', 'Parse error')
                                }
                            except Exception as e:
                                result_map[qa_id] = {'rating': -1, 'reason': f'JSON parse error: {e} (text={text if text else "EMPTY"})'}
                        else:
                            result_map[qa_id] = {'rating': -1, 'reason': 'No text in parts'}
                    else:
                        result_map[qa_id] = {'rating': -1, 'reason': 'No content in candidate'}
                else:
                    result_map[qa_id] = {'rating': -1, 'reason': 'No candidates in response'}
            elif 'error' in result_obj:
                error_msg = result_obj['error'].get('message', 'Unknown error')
                result_map[qa_id] = {'rating': -1, 'reason': f'API error: {error_msg}'}
        except Exception as e:
            log_and_print(logger, f"  Warning: Failed to parse result line: {e}")

    for qa in qa_pairs:
        if qa['id'] not in result_map:
            result_map[qa['id']] = {'rating': -1, 'reason': 'Missing response'}

    success_count = sum(1 for r in result_map.values() if r['rating'] != -1)
    log_and_print(logger, f"  Completed: {success_count} success, {len(result_map)-success_count} errors")

    total_tokens = total_input_tokens + total_output_tokens
    log_and_print(logger, f"  Token Usage - Input: {total_input_tokens:,}, Output: {total_output_tokens:,}, Total: {total_tokens:,}")

    token_usage = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_tokens
    }

    return result_map, token_usage, result_lines


def save_detailed_token_analysis(batch_num, qa_pairs, result_lines, token_usage, output_dir):
    """Save per-request token usage with full prompts and responses for analysis."""
    log_dir = os.path.join(output_dir, "token_analysis")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"batch_{batch_num:04d}_token_details.json")

    results_by_key = {}
    for line in result_lines:
        if not line.strip():
            continue
        try:
            result_obj = json.loads(line)
            key = result_obj.get('key', '')
            results_by_key[key] = result_obj
        except:
            pass

    detailed_entries = []
    for qa in qa_pairs:
        qa_id = qa['id']
        result = results_by_key.get(qa_id, {})

        usage_metadata = {}
        if 'response' in result:
            usage_metadata = result['response'].get('usageMetadata', {})

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
                'rating': None,
                'reason': None,
                'full_response_text': None
            },
            'tokens': {
                'input_tokens': usage_metadata.get('promptTokenCount', 0),
                'output_tokens': usage_metadata.get('candidatesTokenCount', 0),
                'total_tokens': usage_metadata.get('totalTokenCount', 0)
            }
        }

        if 'response' in result:
            response = result['response']
            if 'candidates' in response and len(response['candidates']) > 0:
                candidate = response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        text = parts[0]['text'].strip()
                        entry['response']['full_response_text'] = text

                        try:
                            if text.startswith('```'):
                                lines = text.split('\n')
                                if lines[0].startswith('```'):
                                    lines = lines[1:]
                                if lines and lines[-1].strip() == '```':
                                    lines = lines[:-1]
                                text = '\n'.join(lines).strip()

                            response_json = json.loads(text)
                            entry['response']['rating'] = response_json.get('rating')
                            entry['response']['reason'] = response_json.get('reason')
                        except:
                            pass

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
        print("Usage: python judge_gemini.py <exploded_csv_file>")
        sys.exit(1)

    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        sys.exit(1)

    logger = setup_logger(LOG_FILE)
    log_and_print(logger, "="*80)
    log_and_print(logger, f"Gemini Judge")
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

        existing_rating = str(row.get('Gemini_Judge_Rating', '')).strip()
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

        results, token_usage, result_lines = run_batch_judge(API_KEY, batch_qa_pairs, logger)

        total_stats['num_pairs'] += len(batch_qa_pairs)
        total_stats['num_batches'] += 1
        total_stats['input_tokens'] += token_usage['input_tokens']
        total_stats['output_tokens'] += token_usage['output_tokens']
        total_stats['total_tokens'] += token_usage['total_tokens']

        write_batch_output(start_batch_num + batch_idx, batch_qa_pairs, results, OUTPUT_DIR, logger)

        save_detailed_token_analysis(start_batch_num + batch_idx, batch_qa_pairs, result_lines, token_usage, OUTPUT_DIR)
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
