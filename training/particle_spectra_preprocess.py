import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.decomposition import PCA
from scipy.stats import f
from pybaselines import Baseline

def read_witec_txt(filename):
    spectra = []
    buffer = []
    reading = False
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[SpectrumData]"):
                buffer = []
                reading = True
                continue
            if reading:
                if line.startswith("[") or line == "":
                    if buffer:
                        spectra.append(np.array(buffer, dtype=float))
                    reading = False
                else:
                    buffer.append(line.split())
    if buffer:
        spectra.append(np.array(buffer, dtype=float))
    return spectra

def txt_to_matrix(txt_files):
    spectra = []
    for f in txt_files:
        spectra.extend(read_witec_txt(f))
    x = spectra[0][:, 0]
    intensity_matrix = np.column_stack([
        spec[:, 1] for spec in spectra
    ])
    return x, intensity_matrix.T

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
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return spectra / norms

input_root = "dataset/particle_spectra"
output_root = "processed/preprocessed_particle_spectra"
classes = os.listdir(input_root)
for cls in classes:
    class_input_dir = os.path.join(input_root, cls)
    class_output_dir = os.path.join(output_root, cls)
    os.makedirs(class_output_dir, exist_ok=True)
    txt_files = glob(os.path.join(class_input_dir, "*.txt"))
    if len(txt_files) == 0:
        continue
    raman_shift, spectra = txt_to_matrix(txt_files)
    spectra = pca_t2_qc(spectra, n_components=10)
    spectra = np.array([remove_cosmic_spikes(s) for s in spectra])
    baseline = Baseline()
    spectra = np.array([s - baseline.arpls(s, lam=1e6)[0] for s in spectra])
    spectra = vector_normalize_batch(spectra)
    df = pd.DataFrame(spectra, columns=[f"{int(x)}" for x in raman_shift])
    output_csv = os.path.join(class_output_dir, "CNN_input.csv")
    df.to_csv(output_csv, index=False)