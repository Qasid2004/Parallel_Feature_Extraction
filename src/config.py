# src/config.py
import os


# default config values
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 4))
IMAGE_SIZE = (256, 256) # resize images to this size for consistent features
BINS = 32 # bins for color histograms
HOG_DIM = 128
INPUT_DIR = os.environ.get('INPUT_DIR', 'data/input_images')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'data/output_features')
LOG_PATH = os.environ.get('LOG_PATH', 'logs/run_logs.txt')