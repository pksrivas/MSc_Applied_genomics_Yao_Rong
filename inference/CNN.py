import os
import numpy as np
import torch
import torch.nn as nn
from pybaselines import Baseline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "training", "models", "PE_PET_lipid.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNAdaptivePool2(nn.Module):
    def __init__(self):
        super().__init__()
        k1, k2, k3 = 9, 6, 3

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, k1, stride=k1 // 2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, k2, stride=k2 // 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, k3, stride=k3 // 2), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model = CNNAdaptivePool2().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_mapping"].items()}
reference_shifts = np.asarray(checkpoint["raman_shifts"], dtype=np.float32)
input_length = checkpoint["input_length"]


def load_txt_spectrum(path):
    x, y, reading = [], [], False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("[SpectrumData]"):
                reading = True
                continue

            if reading:
                if line.startswith("[") or not line:
                    break

                p = line.split()

                if len(p) >= 2:
                    x.append(float(p[0]))
                    y.append(float(p[1]))

    if not x:
        raise ValueError(f"No spectrum data found: {path}")

    return np.asarray(x, float), np.asarray(y, float)


def remove_cosmic_spikes(y, threshold=6, m=5):
    y = np.asarray(y, float).copy()
    d = np.diff(y)
    med = np.median(d)
    mad = np.median(np.abs(d - med))

    if mad < 1e-12:
        return y

    z = 0.6745 * (d - med) / mad
    spikes = np.where(np.abs(z) > threshold)[0] + 1
    z = np.pad(np.abs(z), (1, 0))

    for i in spikes:
        w = np.arange(max(0, i - m), min(len(y), i + m + 1))
        good = z[w] < threshold

        if np.any(good):
            y[i] = np.mean(y[w][good])

    return y


def baseline_correct(y):
    return y - Baseline().arpls(y, lam=1e6)[0]


def vector_normalize(y):
    norm = np.linalg.norm(y)
    return y / (norm if norm else 1)


def preprocess_raman(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]

    y = remove_cosmic_spikes(y)
    y = np.interp(reference_shifts, x, y)
    y = baseline_correct(y)
    y = vector_normalize(y).astype(np.float32)

    return reference_shifts, y


def predict_raman_file(path):
    x, y = preprocess_raman(*load_txt_spectrum(path))

    if len(y) != input_length:
        raise ValueError(
            f"Input length {len(y)} does not match model input length {input_length}"
        )

    signal = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.softmax(model(signal), dim=1)
        confidence, pred = torch.max(prob, dim=1)

    return x, y, idx_to_class[pred.item()], confidence.item()