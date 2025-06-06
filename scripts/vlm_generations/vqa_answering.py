"""

Runs VQA answering on a model.
To run, either call run_qa() or run this script with the given parameters.

@Nathan Anderson

"""

import ast
import json
import time
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import scripts.anthropic as anthropic_image_to_text

import data.vqa_prompt.prompt_qa as prompt_qa


DATASET_CSV_PATH = "../../Dataset/DrawEduMath_QA.csv"
# The path to the folder with the dataset images.
# If you don't have the images downloaded, you can set `url=True` when calling `run_qa()`,
# which will send the image URL in the request instead of the base64-encoded image.
IMAGE_FOLDER_PATH = "../../Dataset/Images_w_Problem/"

# -----Claude 3.7 Sonnet parameters (change as necessary)-----
RATE_LIMIT = 50  # The rate limit, in requests per minute.
SLEEP_TIME = 60  # Time to sleep after the rate limit is reached, in seconds

MODEL_NAME = "Claude_3.7Sonnet"  # The name of the model.
MODEL_API_NAME = "claude-3-7-sonnet-20250219"  # The name of the model in the API request you send.
API = anthropic_image_to_text.AnthropicImageToText

# In some cases, requests may be rate limited per-second rather than per-minute.
# If that's the case, set this to the number of seconds to wait between API requests.
# Also set `delay_submissions=True` when calling `run_qa()`.
# This is significantly slower than using `delay_submissions=False`, so only do this if you need to.
SECONDS_PER_REQUEST = 2


def get_questions(qa_json):
    # qa_list = json.loads(qa_json)
    qa_list = ast.literal_eval(qa_json)

    questions = [item["question"] for item in qa_list if "question" in item]
    return questions


def load_qa_json(qa_pairs):
    try:
        qa = json.loads(qa_pairs)
        qa = ast.literal_eval(qa)

        return qa
    except:
        qa = json.loads(qa_pairs)
        return qa


# Process a single row
def process_row_image(image_id, question, url):
    """
    Run the model on a single image and question.
    """
    try:
        api = API(MODEL_API_NAME)
        image_path = image_id if url else IMAGE_FOLDER_PATH + image_id

        system_prompt = prompt_qa.GENERATE_ANSWER_PROMPT
        user_prompt = "Answer the following question: " + str(question)

        response = api.get_response(image_path, system_prompt, user_prompt)
        return {"question": question, "answer": response}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"question": question, "answer": "Error"}


def run_qa(
    df: pd.DataFrame, teacher: bool, gpt4o: bool, claude: bool, delay_submissions: bool = True, url: bool = False
):
    """
    Run the model on QA pairs.

    @param df: The DataFrame containing the QA pairs.
    @param teacher: Whether to run the model on the teacher QA pairs.
    @param gpt4o: Whether to run the model on the GPT4o QA pairs.
    @param claude: Whether to run the model on the Claude QA pairs.
    @param delay_submissions: Whether to delay submissions to the API. This can be useful if an API rate limits
    per-second rather than per-minute.
    @param url: Whether to submit the image as a URL or a base64-encoded string. Default is base64-encoded string (False).
    """

    save_path = "../output/" + MODEL_NAME + "_All_QA_Pairs_Output.csv"
    teacher_col_name = MODEL_NAME + "_Teacher_QA_Pairs"
    gpt4o_col_name = MODEL_NAME + "_GPT4o_QA_Pairs"
    claude_col_name = MODEL_NAME + "_Claude_QA_Pairs"

    qa_pair_names = []

    if teacher:
        qa_pair_names.append(("QA Teacher", teacher_col_name))
        if teacher_col_name not in df:
            df[teacher_col_name] = np.nan
    if gpt4o:
        qa_pair_names.append(("QA GPT4o", gpt4o_col_name))
        if gpt4o_col_name not in df:
            df[gpt4o_col_name] = np.nan
    if claude:
        qa_pair_names.append(("QA Claude", claude_col_name))
        if claude_col_name not in df:
            df[claude_col_name] = np.nan

    request_count = 0
    print(f"Model will be run on: {[q[0] for q in qa_pair_names]}")
    print(f"Delay submissions: {delay_submissions}")
    print(f"Use image URLs instead of base64-encoded images: {url}")
    print(f"Results will be saved to {save_path}")

    reply = input("Are you sure you want to continue? (y/n) ")
    if reply.lower() != "y":
        print("Exiting")
        return

    for qa_col_name, result_col_name in qa_pair_names:
        print(f"Running on {qa_col_name}")
        print(f"Outputting to column {result_col_name}")

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Skip over columns that have already been populated (inference was already run)
            if isinstance(df.loc[i, result_col_name], str):
                continue

            questions = get_questions(row[qa_col_name])

            image_id = row["Image URL"] if url else row["Image Name"]

            qa_pairs = []

            # Using ThreadPoolExecutor for parallel question processing
            with ThreadPoolExecutor() as executor:
                # Asynchronously send requests to the API
                if not delay_submissions:
                    futures = [
                        executor.submit(process_row_image, image_id, question, url) for question in questions
                    ]

                    # Collect results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        qa_pair = future.result()
                        qa_pairs.append(qa_pair)

                        # Increment request count
                        request_count += 1

                        if request_count % 50 == 0:
                            print(f"Saved intermediary df to {save_path}")
                            df.to_csv(save_path, index=False)

                        # If request count reaches the rate limit, pause for a minute
                        if request_count >= RATE_LIMIT:
                            print(f"Rate limit reached, sleeping for {SLEEP_TIME} seconds...")
                            time.sleep(SLEEP_TIME)
                            request_count = 0  # Reset request count after sleeping
                # Don't asynchronously send requests to the API
                else:
                    for question in tqdm(questions):
                        qa_pair = process_row_image(image_id, question, url)
                        qa_pairs.append(qa_pair)

                        request_count += 1

                        time.sleep(SECONDS_PER_REQUEST)

                        # If request count reaches the rate limit, pause for a minute
                        if request_count >= 50:
                            print(f"Saved intermediary df to {save_path}")
                            df.to_csv(save_path, index=False)
                            # print(f"Rate limit reached, sleeping for {SLEEP_TIME} seconds...")
                            # time.sleep(SLEEP_TIME)
                            request_count = 0  # Reset request count after sleeping

            df.loc[i, result_col_name] = json.dumps(qa_pairs)
            # print(qa_pairs)

        df.to_csv(save_path, index=False)


def main():
    df = pd.read_csv(DATASET_CSV_PATH)
    # Uncomment this line if your inference failed partway through.
    # df = pd.read_csv(f"../output/{MODEL_NAME}_All_QA_Pairs.csv")
    run_qa(df, teacher=True, gpt4o=True, claude=True, delay_submissions=True, url=False)


if __name__ == "__main__":
    main()
