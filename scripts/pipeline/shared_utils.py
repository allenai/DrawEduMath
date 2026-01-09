import csv
import json
import ast
import logging
from datetime import datetime


def setup_logger(log_file):
    """Create logger that writes to file with timestamps."""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_and_print(logger, message):
    """Write message to log file and print to console."""
    logger.info(message)
    print(message)


def read_csv_as_dicts(filepath):
    """Load CSV file into list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv_from_dicts(filepath, data, fieldnames):
    """Write list of dictionaries to CSV file."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(data)


def save_checkpoint(filepath, data, fieldnames, logger, checkpoint_num):
    """Save progress checkpoint with logging."""
    write_csv_from_dicts(filepath, data, fieldnames)
    log_and_print(logger, f"  Checkpoint saved: {checkpoint_num} rows")


def get_questions(qa_json):
    """Extract question strings from QA JSON, handling multiple formats."""
    try:
        qa_list = json.loads(qa_json)
        if isinstance(qa_list, str):
            qa_list = ast.literal_eval(qa_list)
    except (json.JSONDecodeError, ValueError, SyntaxError):
        qa_list = ast.literal_eval(qa_json)

    questions = [item['question'] for item in qa_list if 'question' in item]
    return questions
