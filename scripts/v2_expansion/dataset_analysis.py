"""
Perform some rudimentary analyzis on the dataset.

In particular, this script supports the following:
- Distribution of the number of images per problem
- Distribution of the mean scores for each problem
- Distribution of the grade levels across all problems

Uncomment the function calls at the bottom of the script to get what you want.
"""

import matplotlib.pyplot as plt
import pandas as pd

# base_df = pd.read_csv("./Dataset/MathNet57789_Problem_Logs.csv")

# CSV of the v2 sampled results
df = pd.read_csv("./sampled_results.csv")

# Problem bodies for MathNet57789
problem_bodies_df = pd.read_csv("./Dataset/Problem_Bodies.csv")


def number_per_problem():
    """
    Show a histogram of the number of images per problem.
    """
    pids = df["problem_id"].unique()
    counts = [len(df[df["problem_id"] == pid]) for pid in pids]
    plt.figure()
    plt.xlabel("Number of images in problem")
    plt.ylabel("Number of problems")
    plt.hist(counts, bins=20)
    plt.show()


def means():
    """
    Show a histogram of the mean score for each problem.
    """
    scores = []

    for pid in df["problem_id"].unique():
        problem_scores = df[df["problem_id"] == pid]["score"]
        print(pid, len(problem_scores))
        # print(pid, problem_scores.mean())
        scores.append(problem_scores)

    print(f"Number of images: {len(df)}")
    print(f"Mean score: {df['score'].mean()}")

    plt.figure()
    plt.hist(df["score"], bins=20)
    plt.show()
    plt.figure()
    plt.hist(scores, bins=20)
    plt.show()


def grade_levels():
    """
    Show a distribution of the grade levels.
    """
    grade_levels = []

    for _, row in df.iterrows():
        pid = row["problem_id"]
        problem_body = problem_bodies_df[problem_bodies_df["problem_id"] == pid]
        if len(problem_body) > 0:
            grade_levels.append(problem_body.iloc[0]["grade_or_subject"])

    grade_levels = [i for i in grade_levels if "Grade" in i]
    grade_levels.sort()
    plt.figure()
    plt.xlabel("Grade Level")
    plt.ylabel("Number of Images")
    plt.hist(grade_levels, bins=20)
    plt.show()


# grade_levels()
# means()
number_per_problem()
