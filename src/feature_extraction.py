import cv2
import numpy as np
from numba import njit, prange

def load_image(path, size=None):
    # Handles non-ASCII paths on Windows
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def color_histogram(img, bins=32):
    chans = cv2.split(img)
    features = []
    for ch in chans:
        hist = cv2.calcHist([ch], [0], None, [bins], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)

def edge_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    edges = cv2.Canny(gray, 100, 200)
    edges_count = int(np.sum(edges > 0))
    return np.array([mean_mag, std_mag, edges_count], dtype=np.float32)

def hog_descriptor(img, win_size=(64,128), out_dim=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if (w, h) != win_size:
        gray = cv2.resize(gray, win_size, interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor()
    desc = hog.compute(gray)
    desc = desc.flatten()
    if len(desc) >= out_dim:
        return desc[:out_dim].astype(np.float32)
    else:
        pad = np.zeros(out_dim - len(desc), dtype=np.float32)
        return np.concatenate([desc.astype(np.float32), pad])

# Numba accelerated RGB histogram
@njit(parallel=True)
def rgb_hist_numba(img, bins):
    h, w, c = img.shape
    hist = np.zeros((3, bins), dtype=np.int32)
    scale = 256 // bins
    for i in prange(h):
        for j in range(w):
            for k in range(3):
                val = img[i, j, k] // scale
                if val >= bins:
                    val = bins - 1
                hist[k, val] += 1
    out = np.empty(3 * bins, dtype=np.float32)
    total = h * w
    idx = 0
    for ch in range(3):
        for b in range(bins):
            out[idx] = hist[ch, b] / total
            idx += 1
    return out
