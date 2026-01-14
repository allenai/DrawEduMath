"""

Run data analysis on a model's answers. This covers both teacher and synthetic QA.
It calculates the mean, standard deviation, and 95% confidence intervals for each category, as well as for overall results.

@Nathan Anderson

"""

import json
import math

import numpy as np
import pandas as pd


MODEL_NAME = "gpt_4.1"


# Calculate the confidence interval for the given data.
# Default z is  1.96 (95%).
def ci(data: np.ndarray, z: float = 1.96) -> float:
    return z * data.std() / math.sqrt(len(data))


# Parse the LLM score from the given line
def parse_llm_score(line: str):
    j = json.loads(line)

    if isinstance(j, str):
        # If j is a string, there are escaped characters in it, some of which are invalid.
        # To get around this, we extract the rating and reason and manually create a dict ourselves.
        try:
            rating = int(j.split('"rating": ')[1].split(', "')[0])
        except:
            try:
                rating = int(float(j.split('"rating": ')[1].split(",")[0]))
            except:
                try:
                    rating = int(float(j.split(r"\"rating\": ")[1].split(",")[0]))
                except:
                    rating = 0

        try:
            reason = j.split('"reason": "')[1]
        except:
            try:
                reason = j.split(r"\"reason\": \"")[1]
            except:
                reason = ""
        # Remove }" at the end of the string
        reason = reason[: len(reason) - 2]

        j = {"rating": rating, "reason": reason}

    try:
        return j["rating"] / 4.0
    except:
        return None


def parse_bert_rouge(line: str):
    try:
        if not np.isnan(float(line)):
            return float(line)
        else:
            return None
    except:
        return None


# Print the mean, std, and confidence interval for the given data.
def print_stats(values: np.ndarray):
    print(f"\tMean: {values.mean()}")
    print(f"\t Std: {values.std()}")
    confidence_interval = ci(values)
    print(
        f"\t 95%: ({values.mean() - confidence_interval}, {values.mean() + confidence_interval})"
        f" ({confidence_interval})"
    )
    print(f"\t   n: {len(values)}")


def main():
    # -----Teacher-----
    df = pd.read_csv(f"../output/{MODEL_NAME}_Evaluation_QA_Teacher_LLM.csv")
    bert_f1 = df[f"{MODEL_NAME}_BERTScore_f1"]
    bert_f1 = np.array([i for i in map(parse_bert_rouge, bert_f1) if i is not None])

    rougel = df[f"{MODEL_NAME}_ROUGEL"]
    rougel = np.array([i for i in map(parse_bert_rouge, rougel) if i is not None])

    llm = df[df[f"{MODEL_NAME}_LLM_Score"].notnull()][f"{MODEL_NAME}_LLM_Score"]
    llm = np.array([i for i in map(parse_llm_score, llm) if i is not None])

    print("TEACHER:")
    print("BERT F1:")
    print_stats(bert_f1)
    print("ROUGEL:")
    print_stats(rougel)
    print("LLM:")
    print_stats(llm)

    # -----Synthetic-----
    df1 = pd.read_csv(f"../output/{MODEL_NAME}_Evaluation_QA_GPT4o_LLM.csv")
    df2 = pd.read_csv(f"../output/{MODEL_NAME}_Evaluation_QA_Claude_LLM.csv")

    bert_f1 = df1[f"{MODEL_NAME}_BERTScore_f1"].tolist() + df2[f"{MODEL_NAME}_BERTScore_f1"].tolist()
    bert_f1 = np.array([i for i in map(parse_bert_rouge, bert_f1) if i is not None])

    rougel = np.array(df1[f"{MODEL_NAME}_ROUGEL"].tolist() + df2[f"{MODEL_NAME}_ROUGEL"].tolist())
    rougel = np.array([i for i in map(parse_bert_rouge, rougel) if i is not None])

    llm = (
        df1[df1[f"{MODEL_NAME}_LLM_Score"].notnull()][f"{MODEL_NAME}_LLM_Score"].tolist()
        + df2[df2[f"{MODEL_NAME}_LLM_Score"].notnull()][f"{MODEL_NAME}_LLM_Score"].tolist()
    )
    print(
        f"Discarded {len([i for i in map(parse_llm_score, llm) if i is None])} LLM scores (reason: could not parse)."
    )
    llm = np.array([i for i in map(parse_llm_score, llm) if i is not None])

    print("\nSYNTHETIC:")
    print("BERT F1:")
    print_stats(bert_f1)
    print("ROUGEL:")
    print_stats(rougel)
    print("LLM:")
    print_stats(llm)

    print("\n\n----------PER-CATEGORY----------")
    print("\n-----TEACHER-----")
    per_category([df])
    print("\n-----SYNTHETIC-----")
    per_category([df1, df2])


def per_category(dfs: list[pd.DataFrame]):
    with open("../data/question_to_category_second_formatted.json") as f:
        j: dict[str, str] = json.load(f)

    categories = sorted(list(set(j.values())))
    print(categories)

    bert_f1s: dict[str, list[float]] = {
        "Image creation and medium": [],
        "Problem solving steps, strategy, and solution": [],
        "Counting content": [],
        "Low-level characteristics, composition, and positioning": [],
        "Correctness and errors": [],
        "Other": [],
        "Writing and labels": [],
        "Higher-level understanding of math concepts": [],
    }

    rouges: dict[str, list[float]] = {
        "Image creation and medium": [],
        "Problem solving steps, strategy, and solution": [],
        "Counting content": [],
        "Low-level characteristics, composition, and positioning": [],
        "Correctness and errors": [],
        "Other": [],
        "Writing and labels": [],
        "Higher-level understanding of math concepts": [],
    }

    llms: dict[str, list[float]] = {
        "Image creation and medium": [],
        "Problem solving steps, strategy, and solution": [],
        "Counting content": [],
        "Low-level characteristics, composition, and positioning": [],
        "Correctness and errors": [],
        "Other": [],
        "Writing and labels": [],
        "Higher-level understanding of math concepts": [],
    }

    for df in dfs:
        for i, row in df.iterrows():
            question = row["Question"]
            if isinstance(question, str):
                question = question.strip()
                if question not in j:
                    print(f"Question not found: {question}")
                else:
                    category = j[question]
                    if (bert := parse_bert_rouge(row[f"{MODEL_NAME}_BERTScore_f1"])) is not None and not np.isnan(
                        bert
                    ):
                        bert_f1s[category].append(bert)
                    else:
                        print(f"Could not parse BERTScore: {row[f'{MODEL_NAME}_BERTScore_f1']}")

                    if (rouge := parse_bert_rouge(row[f"{MODEL_NAME}_ROUGEL"])) is not None and not np.isnan(
                        rouge
                    ):
                        rouges[category].append(rouge)
                    else:
                        print(f"Could not parse ROUGEL: {row[f'{MODEL_NAME}_ROUGEL']}")

                    if (llm := parse_llm_score(row[f"{MODEL_NAME}_LLM_Score"])) is not None and not np.isnan(llm):
                        llms[category].append(llm)
                    else:
                        print(f"Could not parse LLM Score: {row[f'{MODEL_NAME}_LLM_Score']}")

    for category in categories:
        print(f"\n{category}:")
        print("BERT F1:")
        print_stats(np.array(bert_f1s[category]))
        print("ROUGEL:")
        print_stats(np.array(rouges[category]))
        print("LLM:")
        print_stats(np.array(llms[category]))


if __name__ == "__main__":
    # per_category([pd.read_csv(f"../output/{MODEL_NAME}_Evaluation_QA_Teacher_LLM.csv")])
    main()
