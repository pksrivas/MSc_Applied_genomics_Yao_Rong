import os
import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.stats import f
from sklearn.decomposition import PCA

def pca_t2_qc(X, n_components=10):
    n = X.shape[0]
    A = min(n_components, n - 2)
    pca = PCA(n_components=A)
    scores = pca.fit_transform(X - X.mean(axis=0))
    T2 = np.sum(scores**2 / pca.explained_variance_, axis=1)
    limit = (A * (n - 1) * (n + 1) / (n * (n - A)) * f.ppf(0.99, A, n - A))
    return X[T2 <= limit]

def remove_cosmic_spikes(spectrum, threshold=6, m=5):
    y = np.asarray(spectrum, dtype=float).copy()
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    if mad < 1e-12:
        return y
    z = 0.6745 * (d - np.median(d)) / mad
    spikes = np.where(np.abs(z) > threshold)[0] + 1
    z_pad = np.pad(np.abs(z), (1, 0))
    for i in spikes:
        w = np.arange(max(0, i-m), min(len(y), i+m+1))
        good = z_pad[w] < threshold
        if np.any(good):
            y[i] = np.mean(y[w][good])
    return y

def vector_normalize_batch(spectra):
    spectra = np.asarray(spectra, dtype=float)
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return spectra / norms

input_root = "dataset/MPs_in_cells_spectra"
output_root = "processed/preprocessed_MPs_in_cells_spectra"
target_shifts = np.array([
    532, 535, 537, 540, 543, 546, 548, 551, 554, 557,
    559, 562, 565, 568, 570, 573, 576, 579, 581, 584,
    587, 590, 592, 595, 598, 601, 603, 606, 609, 612,
    614, 617, 620, 622, 625, 628, 631, 633, 636, 639,
    642, 644, 647, 650, 652, 655, 658], dtype=float)
for root, dirs, files in os.walk(input_root):
    for file in files:
        if not file.lower().endswith(".txt"):
            continue
        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.read_csv(input_path, sep="\t", header=None)
        raman_shift = (df.iloc[:, 0] .values .astype(float))
        spectra = (df.iloc[:, 1:] .values .T)
        spectra = pca_t2_qc(spectra, n_components=10)
        spectra_despiked = np.array([remove_cosmic_spikes(s) for s in spectra])
        baseline = Baseline()
        spectra_baseline_corrected = np.array([s - baseline.arpls(s, lam=1e6)[0] for s in spectra])
        spectra_normalized = (vector_normalize_batch(spectra_baseline_corrected))
        indices = [np.argmin(np.abs(raman_shift - s)) for s in target_shifts]
        sampled_spectra = (spectra_normalized[:, indices])
        result = np.vstack([target_shifts, sampled_spectra])
        pd.DataFrame(result).to_csv(output_path, index=False, header=False)