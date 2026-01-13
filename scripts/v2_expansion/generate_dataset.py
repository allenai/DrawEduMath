import numpy as np
import pandas as pd

# The target number of images to have, in total, per problem.
TARGET_PER_PROBLEM = 20

# Biases for each score (the probability the score will be chosen)
SCORE_BIASES = {0: 0, 1: 0.3, 2: 0.4, 3: 0.3, 4: 0.0}


def main():
    # MathNet57789 CSV
    all_df = pd.read_csv("./Dataset/MathNet57789_Problem_Logs.csv")

    # DrawEduMath CSV
    df = pd.read_csv("../Dataset/DrawEduMath_QA.csv")

    # All images in the v1 dataset
    dataset_df = all_df[all_df["image_name"].isin(df["Image Name"])]

    # All images in MathNet57789 that are not in v1
    all_df = all_df[~all_df["image_name"].isin(df["Image Name"])]

    all_pids: list[int] = all_df["problem_id"].unique()  # type: ignore

    results = []

    for pid in all_pids:
        images = select_images(dataset_df, all_df, pid)
        results.append(images)

    results = pd.concat(results)
    print(len(results))
    final_df = pd.concat([results, dataset_df])
    print(len(final_df))
    final_df.to_csv("./sampled_results.csv", index=False)


def select_images(dataset_df, all_df, pid) -> list[str]:
    """
    Select images for the given problem ID.
    """
    dataset_images = dataset_df[dataset_df["problem_id"] == pid]

    # All the images in the problem
    all_images = all_df[all_df["problem_id"] == pid]

    # Remove images with a score of 0 or 4, since we want to lower the overall score of the dataset.
    all_images = all_images[all_images["score"] != 4]
    all_images = all_images[all_images["score"] != 0]

    # The number of images to select for the problem in order to pad out v2 to 20 images.
    n_images = TARGET_PER_PROBLEM - len(dataset_images)

    biases = [SCORE_BIASES[entry["score"]] for _, entry in all_images.iterrows()]
    biases = biases / np.sum(biases)
    return all_images.sample(n=min(n_images, len(all_images)), weights=biases)


def average_score(df, pid: int) -> float:
    """
    Given a problem ID, find the average score for that problem's answers in the given df.
    """
    entries = df[df["problem_id"] == pid]
    return entries["score"].mean()  # type: ignore


if __name__ == "__main__":
    main()
