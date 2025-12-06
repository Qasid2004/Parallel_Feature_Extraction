from utils import list_images, save_features_csv, log
from parallel_processing import run_single, run_multiprocessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--mode', type=str, default='single', choices=['single','multiprocessing'])
parser.add_argument('--show_gui', action='store_true')
args = parser.parse_args()

image_paths = list_images(args.input_dir)

if args.mode == 'multiprocessing':
    rows = run_multiprocessing(image_paths, workers=4, show_img=args.show_gui)
else:
    rows = run_single(image_paths, show_img=args.show_gui)

# Columns = image_name + features
# Columns = image_name + features
if rows:
    # Example feature sizes (adjust based on your feature_extractor):
    # edge_stats = 3 (mean_mag, std_mag, edges_count)
    # color_histogram = 96 (32 bins x 3 channels)
    # HOG descriptor = 128
    # RGB histogram via numba = 96 (32 bins x 3 channels)
    
    edge_cols = ['edge_mean', 'edge_std', 'edge_count']
    color_cols = [f'color_{c}_{i}' for c in ['r','g','b'] for i in range(32)]
    hog_cols = [f'hog_{i}' for i in range(128)]
    rgb_cols = [f'rgb_{c}_{i}' for c in ['r','g','b'] for i in range(32)]
    
    cols = ['image'] + edge_cols + color_cols + hog_cols + rgb_cols

    save_features_csv([(r[0], *r[1]) for r in rows], cols, args.out)
    log(f"Features saved to {args.out}")
else:
    log("No features extracted.")
