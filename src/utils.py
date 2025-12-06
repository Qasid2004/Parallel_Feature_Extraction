import os
from glob import glob
import pandas as pd
import time

def list_images(folder, exts=("jpg", "jpeg", "png", "bmp")):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"**/*.{ext}"), recursive=True))
    files = sorted(files)
    return files

def save_features_csv(rows, cols, out_file):
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_file, index=False)

def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def log(msg, path='logs/run_logs.txt'):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        f.write(f"[{timestamp()}] {msg}\n")
