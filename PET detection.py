import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_PATH = "training/models/PET_detection.pth"
SCAN_PATH = "training/processed/preprocessed_MPs_in_cells_spectra/PET_THP-1/Area 2/PET_THP-1_Scan 1.csv"
PET_INDEX_PATH = "training/processed/PET_indices.csv"
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleCNN(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                for k in kernels
            ]
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        features = [branch(x).flatten(1) for branch in self.branches]
        return self.classifier(torch.cat(features, dim=1)).flatten()


def load_scan(path, model_shifts):
    data = pd.read_csv(path, header=None)
    shifts = pd.to_numeric(data.iloc[0], errors="coerce").to_numpy()

    indices = np.array(
        [np.argmin(np.abs(shifts - shift)) for shift in model_shifts]
    )

    if not np.allclose(shifts[indices], model_shifts, rtol=0, atol=1e-4):
        raise ValueError("Scan Raman shifts do not match the model.")

    X = (
        data.iloc[1:, indices]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float32)
    )

    if not np.isfinite(X).all():
        raise ValueError("Scan contains non-numeric or missing spectral values.")

    return X


def predict(model, X):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[start : start + BATCH_SIZE])
            batch = batch.float().to(DEVICE).unsqueeze(1)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())

    return np.concatenate(probabilities)


def main():
    output_dir = os.path.dirname(PET_INDEX_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model_shifts = np.asarray(
        checkpoint["raman_shifts"],
        dtype=np.float32,
    )

    X = load_scan(SCAN_PATH, model_shifts)
    input_length = checkpoint["input_length"]

    if X.shape[1] != input_length:
        raise ValueError(
            f"Expected {input_length} Raman features, got {X.shape[1]}."
        )

    model = MultiScaleCNN(tuple(checkpoint["kernels"])).to(DEVICE)
    model(torch.zeros(1, 1, input_length, device=DEVICE))
    model.load_state_dict(checkpoint["model_state_dict"])

    probabilities = predict(model, X)
    predictions = probabilities >= checkpoint.get("threshold", 0.5)
    pet_indices = np.flatnonzero(predictions) + 1

    pd.DataFrame(
        {
            "Spectrum_ID": pet_indices,
            "PET_Probability": probabilities[predictions],
        }
    ).to_csv(PET_INDEX_PATH, index=False)

    print(f"Total spectra: {len(X)}")
    print(f"Predicted PET spectra: {len(pet_indices)}")
    print(f"PET indices saved to: {PET_INDEX_PATH}")


if __name__ == "__main__":
    main()