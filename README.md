# Parallel Feature Extraction from Images

## Overview
This project extracts multiple image features (edges, color histograms, HOG, and RGB histograms) from a folder of images. Features can be extracted **single-threaded** or **using multiprocessing** for faster performance. Output is saved as a CSV file.

---

## Features Extracted

1. **Edge Statistics**: Mean, Standard Deviation, Edge Count  
2. **Color Histogram**: RGB channels, 32 bins each  
3. **HOG Descriptor**: Histogram of Oriented Gradients (128 dimensions)  
4. **RGB Histogram (Numba accelerated)**: RGB channels, 32 bins each  

---

## Requirements

- Python 3.8+  
- Libraries:

opencv-python
numpy
pandas
tqdm
numba


---

## Installation & Run Guide


1. Download the Project: 
git clone ""
cd Parallel-Feature-Extraction

2. Create Virtual Environment (Cross-Platform):

- Windows (CMD / PowerShell)
python -m venv venv
venv\Scripts\activate

- Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install Required Libraries

Use this single universal command:
pip install -r requirements.txt


4. Running the Program:

Multiprocessing mode (recommended):
python src/main.py --input_dir data/input_images --mode multiprocessing --out data/output_features/features.csv


## Usage


# Single-threaded
python3 src/main.py --input_dir data/input_images --mode single --out data/output_features/features.csv

# Multi-threaded (faster)
python3 src/main.py --input_dir data/input_images --mode multiprocessing --out data/output_features/features.csv

# Optional: show GUI preview of images
python3 src/main.py --input_dir data/input_images --mode multiprocessing --out data/output_features/features.csv --show_gui

Folder Structure

Parallel-Feature-Extraction/
├── data/
│   ├── input_images/       # images to process
│   └── output_features/    # CSV output
├── logs/
│   └── run_logs.txt        # log file
├── src/
│   ├── main.py
│   ├── feature_extraction.py
│   ├── parallel_processing.py
│   ├── utils.py
│   └── config.py
├── requirements.txt
└── README.md
