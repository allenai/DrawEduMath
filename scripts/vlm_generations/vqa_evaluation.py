"""

Runs VQA evaluation (BERTScore, ROUGEL, LLM Score) on a model's VQA answers.
This is intended to be run on the output of vqa_answering.py.
Final results are saved to ../output/<MODEL_NAME>_Evaluation_<QA Pair Type>_LLM.csv

@Nathan Anderson

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import scripts.vlm_generations.vqa_score_generation as vqa_score_generation
import scripts.vlm_generations.text_to_text_generation as text_to_text

import data.vqa_prompt.prompt_qa as prompt_qa


# Set this to the name of the model you want to evaluate.
MODEL_NAME = "gpt_4.1"


def load_qa_json(qa_pairs):
    qa_list = json.loads(qa_pairs)

    return sorted(qa_list, key=lambda x: x["question"])


def evaluate_answers(i, row, model):
    together_api = text_to_text.TogetherTextToText()

    question = row["Question"]
    answer = row["Answer"]
    model_a = row[model]
    user_prompt = f"Question: {question}, Answer 1: {answer}, Answer 2: {model_a}"

    response = together_api.get_response(prompt_qa.EVALUATE_ANSWER_PROMPT, user_prompt)

    try:
        response = json.loads(response)
        return i, response

    except Exception as e:
        print(f"Error: {e}, index: {i}, response: {response}")
    return i, response


def evaluate(df: pd.DataFrame, teacher: bool = True, gpt4o: bool = True, claude: bool = True):
    teacher_col_name = MODEL_NAME + "_Teacher_QA"
    gpt4o_col_name = MODEL_NAME + "_GPT4o_QA"
    claude_col_name = MODEL_NAME + "_Claude_QA"

    qa_pair_names = []
    if teacher:
        qa_pair_names.append(("QA Teacher", teacher_col_name))
    if gpt4o:
        qa_pair_names.append(("QA GPT4o", gpt4o_col_name))
    if claude:
        qa_pair_names.append(("QA Claude", claude_col_name))

    print(f"Model: {MODEL_NAME}")
    print(f"Evaluation will be run on: {[q[0] for q in qa_pair_names]}")

    for col_name, result_col_name in qa_pair_names:
        df[col_name] = df[col_name].apply(load_qa_json)
        df[result_col_name] = df[result_col_name + "_Pairs"].apply(load_qa_json)

    exploded_dfs = []
    for col_name, result_col_name in qa_pair_names:
        exploded_dfs.append(df.explode([col_name, result_col_name]).reset_index(drop=True))

    vqa_scorer = vqa_score_generation.VQAScorer()

    for (col_name, result_col_name), df_exploded in zip(qa_pair_names, exploded_dfs):
        save_path = "../output/" + MODEL_NAME + f"_Evaluation_{col_name.replace(' ', '_')}.csv"

        print(f"Running on {col_name}")
        for i, row in tqdm(df_exploded.iterrows(), total=len(df_exploded)):
            # Skip over columns that have already been populated
            if isinstance(df_exploded.loc[i, result_col_name], str):
                continue

            df_exploded.loc[i, "Question"] = row[col_name]["question"]
            df_exploded.loc[i, "Answer"] = row[col_name]["answer"]
            df_exploded.loc[i, f"{MODEL_NAME}_Answer"] = row[result_col_name]["answer"]

            # BERTScore
            bertscore = vqa_scorer.get_bert_score(
                [df_exploded.loc[i, "Answer"]], [df_exploded.loc[i, f"{MODEL_NAME}_Answer"]]
            )

            df_exploded.loc[i, f"{MODEL_NAME}_BERTScore"] = json.dumps(bertscore)
            df_exploded.loc[i, f"{MODEL_NAME}_BERTScore_f1"] = np.mean(bertscore["f1"])

            # ROUGE
            rouge = vqa_scorer.get_rouge_score(
                [df_exploded.loc[i, "Answer"]], [df_exploded.loc[i, f"{MODEL_NAME}_Answer"]]
            )

            df_exploded.loc[i, f"{MODEL_NAME}_ROUGE"] = json.dumps(rouge)
            df_exploded.loc[i, f"{MODEL_NAME}_ROUGEL"] = rouge["rougeL"]

            if i % 1000 == 0:
                df_exploded.to_csv(save_path, index=False)
                print(f"Saved intermediary results to {save_path}")

        df_exploded.to_csv(save_path, index=False)

        # LLM
        print("Running LLM scores...")
        save_path = "../output/" + MODEL_NAME + f"_Evaluation_{col_name.replace(' ', '_')}_LLM.csv"
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(evaluate_answers, i, row, f"{MODEL_NAME}_Answer"): i
                for i, row in df_exploded.iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                i, response = future.result()

                try:
                    rating = int(response.split('"rating": ')[1].split(",")[0])
                    reason = response.split('"reason": "')[1]
                    # Remove }" at the end of the string
                    reason = reason[: len(reason) - 2]

                    df_exploded.at[i, f"{MODEL_NAME}_LLM_Score"] = json.dumps({"rating": rating, "reason": reason})
                except:
                    df_exploded.at[i, f"{MODEL_NAME}_LLM_Score"] = json.dumps(response)
                if i % 1000 == 0:
                    df_exploded.to_csv(save_path, index=False)
                    print(f"Saved intermediary results to {save_path}")

        df_exploded.to_csv(save_path, index=False)


def main():
    # df = pd.read_csv(f"../output/Qwen2.5_VL_72b_Instruct_Evaluation_QA_Teacher_LLM.csv")
    df = pd.read_csv(f"../output/{MODEL_NAME}_All_QA_Pairs_Output.csv")
    evaluate(df, teacher=True, gpt4o=True, claude=True)


if __name__ == "__main__":
    main()
