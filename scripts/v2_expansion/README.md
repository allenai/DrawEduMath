This folder contains the scripts that were used to expand the DrawEduMath dataset for its v2 release.

`download_images.py`: Downloads the MathNet57789 images, since they aren't included when you download the dataset from HuggingFace.

`generate_dataset.py`: Selecting / sampling the images to be used in v2.

`dataset_analysis.py`: Some simple data analysis on the generated dataset (eg distribution of scores).

`check.py`: A PyQt5 tool for manually checking sampled images for PII.

The scripts take arguments through constants at the top of the file or at the top of main().

