import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import csv
import time
import logging
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import GENERATE_ANSWER_PROMPT

load_dotenv()

AVAILABLE_MODELS = {
    "gemini-2.5-pro-preview-03-25": {
        "api_name": "gemini-2.5-pro-preview-03-25",
        "display_name": "Gemini_2.5_Pro_Preview_03_25",
        "csv_name": "gemini_2.5_pro_preview_03_25"
    },
    "gemini-2.5-pro": {
        "api_name": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "csv_name": "gemini_2.5_pro"
    },
    "gemini-pro-2.5-preview": {
        "api_name": "gemini-2.5-pro-preview",
        "display_name": "Gemini Pro 2.5 Preview",
        "csv_name": "gemini_pro_2.5_preview"
    },
    "gemini-2.0-flash": {
        "api_name": "gemini-2.0-flash",
        "display_name": "Gemini Flash 2.0",
        "csv_name": "gemini_flash_2.0"
    },
    "gemini-3-pro-preview": {
        "api_name": "gemini-3-pro-preview",
        "display_name": "Gemini 3 Pro Preview",
        "csv_name": "gemini_3_pro_preview"
    }
}

SELECTED_MODEL = "gemini-3-pro-preview"

MODEL_CONFIG = AVAILABLE_MODELS[SELECTED_MODEL]
MODEL_NAME = MODEL_CONFIG["api_name"]
MODEL_TAG = MODEL_CONFIG["display_name"]
CSV_NAME = MODEL_CONFIG['csv_name']
INPUT_CSV = f"../../../output/{CSV_NAME}.csv"
OUTPUT_CSV = f"../../../output/{CSV_NAME}.csv"
IMAGE_FOLDER = "../../../data/AllImages/Resized_Merged_Problem_Images"
LOG_DIR = f"../../../logs/{CSV_NAME}"
LOG_FILE = f"{LOG_DIR}/generation.log"

RATE_LIMIT = 150
SLEEP_TIME = 60
SAVE_INTERVAL = 10


def setup_logger():
    """Configure logging to file and console, suppressing verbose HTTP logs."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('google.auth').setLevel(logging.WARNING)
    logging.getLogger('google.api_core').setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def read_csv_as_dicts(filepath):
    """Load CSV file into list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_from_dicts(filepath, data, fieldnames):
    """Write list of dictionaries to CSV file."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def process_row(row, model, logger):
    """Generate model answer for a single question using vision API."""
    try:
        image_id = row["Image Name"]
        question = row["Question"]
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        image = Image.open(image_path)

        user_prompt = "Answer the following question: " + str(question)
        full_prompt = GENERATE_ANSWER_PROMPT + " " + user_prompt

        response = model.generate_content([full_prompt, image])

        return response.text
    except Exception as e:
        error_msg = str(e).lower()
        if 'rate' in error_msg or 'quota' in error_msg or '429' in error_msg:
            logger.info(f"  Real rate limit hit, sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            try:
                response = model.generate_content([full_prompt, image])
                return response.text
            except Exception as retry_e:
                logger.error(f"Error after retry: {retry_e}")
                return "Error"
        logger.error(f"Error processing row: {e}")
        return "Error"


def run_generation(data, teacher: bool, gpt4o: bool, claude: bool, logger):
    """Generate answers for all unanswered questions, with checkpointing and rate limiting."""

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(MODEL_NAME)

    fieldnames = list(data[0].keys()) if data else []

    if "Model Answer" not in fieldnames:
        fieldnames.append("Model Answer")
        for row in data:
            row["Model Answer"] = ""

    qa_types_to_process = []
    if teacher:
        qa_types_to_process.append("teacher")
    if gpt4o:
        qa_types_to_process.append("gpt4o")
    if claude:
        qa_types_to_process.append("claude")

    rows_to_process = []
    for i, row in enumerate(data):
        if row.get("Model Answer", "").strip() == "" and row["QA Type"] in qa_types_to_process:
            rows_to_process.append((i, row))

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Processing QA Types: {qa_types_to_process}")
    logger.info(f"Total rows to process: {len(rows_to_process)}")
    logger.info(f"Output: {OUTPUT_CSV}")

    request_count = 0
    processed_count = 0

    for idx, (data_idx, row) in enumerate(rows_to_process):
        processed_count += 1

        if processed_count % 10 == 0:
            logger.info(f"Progress: {processed_count}/{len(rows_to_process)} rows ({100*processed_count//len(rows_to_process)}%)")

        answer = process_row(row, model, logger)
        data[data_idx]["Model Answer"] = answer

        request_count += 1

        if request_count % SAVE_INTERVAL == 0:
            logger.info(f"  Checkpoint saved at {request_count} requests")
            write_csv_from_dicts(OUTPUT_CSV, data, fieldnames)

    write_csv_from_dicts(OUTPUT_CSV, data, fieldnames)
    logger.info(f"Generation complete")


def main():
    logger = setup_logger()
    logger.info("="*80)
    logger.info(f"Starting Google VQA Generation - {MODEL_NAME}")
    logger.info("="*80)

    data = read_csv_as_dicts(INPUT_CSV)
    logger.info(f"Loaded dataset: {INPUT_CSV} ({len(data)} rows)")

    run_generation(data, True, True, True, logger)

    logger.info("="*80)
    logger.info("Generation complete")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logger = setup_logger()
        logger.error("="*80)
        logger.error("FATAL ERROR")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        raise
