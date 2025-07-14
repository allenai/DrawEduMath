"""
Checking images in the dataset to make sure they don't have PII.

This is a GUI app made with PyQt5. It displays images so the user can check if
they have PII.

The console will print the current image index in the CSV. If an image cannot be
loaded, it will be skipped.

If the output CSV (./sampled_results.csv) already exists when the program is
run, this script will read the CSV and skip any images that are in it.

The output CSV is NOT autosaved. You need to manually save it by pressing s.

Controls:
- Space: Next image
- x: The current image has PII, remove it from the CSV
- s: Save (there are no auto saves)
- b: Previous image
"""

import os

from pathlib import Path

from PyQt5.QtGui import QKeyEvent, QPixmap
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

# Path to the MathNet57789 folder
DATASET_PATH = Path("./Dataset")

# Path to the MathNet57789 problem logs CSV
PROBLEM_LOGS_PATH = DATASET_PATH / "MathNet57789_Problem_Logs.csv"

# Path to the folder containing the MathNet57789 images
IMAGES_PATH = DATASET_PATH / "Images"

# The name of the CSV to save results to.
OUTPUT_CSV_PATH = Path("./images_sampled.csv")

# images_df = pd.read_csv(PROBLEM_LOGS_PATH)
images_df = pd.read_csv("./sampled_results.csv")
base_df = pd.read_csv("../Dataset/DrawEduMath_QA.csv")
images_df = images_df[~images_df["image_name"].isin(base_df["Image Name"])]

if os.path.exists(OUTPUT_CSV_PATH):
    output_df = pd.read_csv(OUTPUT_CSV_PATH)
else:
    output_df = pd.DataFrame(columns=["image_name"])

images_df = images_df[~images_df["image_name"].isin(output_df["image_name"])]
good = output_df["image_name"].unique().tolist()
print(len(good))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.i = 0
        self.label = QLabel()
        self.setCentralWidget(self.label)

        self._next_image()
        # self._load_image("./Dataset/Images/39051447-b610-493c-9572-6c71abbe23a5.jpeg")

    def keyPressEvent(self, a0) -> None:
        if isinstance(a0, QKeyEvent):
            if a0.text() == " ":
                image_name = images_df.loc[self.i, "image_name"]
                if image_name not in good:
                    good.append(image_name)
                self._next_image()
                print(f"\x1b[K\r{self.i}", end="")
            if a0.text() == "x":
                print(f"\n{images_df.loc[self.i, 'image_name']}")
                self._next_image()
                print(f"\x1b[K\r{self.i}", end="")
            if a0.text() == "s":
                df = pd.DataFrame(good, columns=["image_name"])
                df.to_csv(OUTPUT_CSV_PATH, index=False)
            if a0.text() == "b":
                self._prev_image()
                print(f"\x1b[K\r{self.i}", end="")

    def _get_image_path(self, i):
        return str(IMAGES_PATH / images_df.loc[i, "image_name"])

    def _next_image(self):
        while self.i < 57789:
            self.i += 1
            try:
                return self._load_image(self._get_image_path(self.i))
            except:
                print(f"Skipping {self.i}")

    def _prev_image(self):
        orig_i = self.i
        while self.i > 0:
            self.i -= 1
            try:
                return self._load_image(self._get_image_path(self.i))
            except:
                print(f"Skipping {self.i}")

        self.i = orig_i
        return self._load_image(self._get_image_path(orig_i))

    def _load_image(self, path):
        self.setWindowTitle(path)
        pixmap = QPixmap(path)
        self.label.setPixmap(pixmap)


app = QApplication([])

window = MainWindow()
window.show()

app.exec()
