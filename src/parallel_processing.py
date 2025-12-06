from multiprocessing import Pool
from feature_extraction import load_image, color_histogram, edge_stats, hog_descriptor, rgb_hist_numba
import numpy as np
from tqdm import tqdm
import cv2

def _process_one(args):
    path, resize, show_img = args
    try:
        img = load_image(path, size=resize)
    except Exception as e:
        print(f"Failed {path}: {e}")
        return None

    # Compute features
    ch = color_histogram(img)
    ed = edge_stats(img)
    hog = hog_descriptor(img)
    rgb_hist = rgb_hist_numba(img, bins=32)
    feats = np.concatenate([ch, ed, hog, rgb_hist])

    # Optional GUI preview
    if show_img:
        cv2.imshow("Preview", img)
        cv2.waitKey(200)  # show each image 200 ms
    return (path, feats.astype(float))

def run_multiprocessing(image_paths, workers=4, resize=(256,256), show_img=False):
    inputs = [(p, resize, show_img) for p in image_paths]
    rows = []
    with Pool(processes=workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_one, inputs), total=len(inputs)):
            if res is None:
                continue
            path, feats = res
            rows.append((path, feats))
    if show_img:
        cv2.destroyAllWindows()
    return rows

def run_single(image_paths, resize=(256,256), show_img=False):
    rows = []
    for p in image_paths:
        res = _process_one((p, resize, show_img))
        if res is None:
            continue
        rows.append(res)
    if show_img:
        cv2.destroyAllWindows()
    return rows
