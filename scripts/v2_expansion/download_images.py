"""
Downloads all the images from the dataset.
"""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Path to the MathNet57789 dataset folder
DATASET_PATH = Path("./Dataset")

# Path to the MathNet57789 problem logs CSV
PROBLEM_LOGS_PATH = DATASET_PATH / "MathNet57789_Problem_Logs.csv"

# Path to the folder to save the downloaded images in
OUTPUT_PATH = DATASET_PATH / "Images"

df = pd.read_csv(PROBLEM_LOGS_PATH)

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_url = row["ImageURL"]
    image_name = row["image_name"]

    r = requests.get(image_url)
    open(OUTPUT_PATH / image_name, "wb").write(r.content)
