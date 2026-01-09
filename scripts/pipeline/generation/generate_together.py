import os
import sys
import csv
import time
import base64
import logging
from together import Together
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import GENERATE_ANSWER_PROMPT

load_dotenv()

AVAILABLE_MODELS = {
    "llama-4-scout": {
        "api_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "display_name": "Llama 4 Scout",
        "csv_name": "llama_4_scout"
    }
}

SELECTED_MODEL = "llama-4-scout"

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


def process_row(row, client, logger):
    """Generate model answer for a single question using vision API."""
    try:
        image_id = row["Image Name"]
        question = row["Question"]
        image_path = os.path.join(IMAGE_FOLDER, image_id)

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        user_prompt = "Answer the following question: " + str(question)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": GENERATE_ANSWER_PROMPT},
                {"role": "user", "content": [
                    {
                        "type": "text", "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]}
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing row: {e}")
        return "Error"


def run_generation(data, teacher: bool, gpt4o: bool, claude: bool, logger):
    """Generate answers for all unanswered questions, with checkpointing and rate limiting."""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

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

        answer = process_row(row, client, logger)
        data[data_idx]["Model Answer"] = answer

        request_count += 1

        if request_count % SAVE_INTERVAL == 0:
            logger.info(f"  Checkpoint saved at {request_count} requests")
            write_csv_from_dicts(OUTPUT_CSV, data, fieldnames)

        if request_count >= RATE_LIMIT:
            logger.info(f"  Rate limit reached, sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)
            request_count = 0

    write_csv_from_dicts(OUTPUT_CSV, data, fieldnames)
    logger.info(f"Generation complete")


def main():
    logger = setup_logger()
    logger.info("="*80)
    logger.info(f"Starting Together AI VQA Generation - {MODEL_NAME}")
    logger.info("="*80)

    data = read_csv_as_dicts(INPUT_CSV)
    logger.info(f"Loaded dataset: {INPUT_CSV} ({len(data)} rows)")

    run_generation(data, True, True, True, logger)

    logger.info("="*80)
    logger.info("Generation complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()
