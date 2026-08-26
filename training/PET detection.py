import glob
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


ROOT = "processed/PET detection"
MODEL_PATH = "models/PET_detection.pth"
CURVE_PATH = "processed/PET_detection_ROC_PR_Curves.png"
CM_PATH = "processed/PET_detection_Confusion_Matrix.png"
METRICS_PATH = "processed/PET_detection_Classification_Metrics.png"

RAMAN_MIN, RAMAN_MAX = 530, 660
SEED, EPOCHS = 42, 20
BATCH_SIZE, LR = 512, 1e-3
AUGMENT_RATIO, THRESHOLD = 0.1, 0.5
KERNELS = (3, 7, 11)
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(root, raman_min=RAMAN_MIN, raman_max=RAMAN_MAX):
    X, y, reference_shifts = [], [], None

    for folder, label in [("THP-1_background", 0), ("PET", 1)]:
        for file in sorted(glob.glob(os.path.join(root, folder, "*.csv"))):
            data = pd.read_csv(file, header=None)
            shifts = pd.to_numeric(data.iloc[0], errors="coerce").to_numpy()
            mask = np.isfinite(shifts) & (shifts >= raman_min) & (shifts <= raman_max)

            if not mask.any():
                raise ValueError(
                    f"No Raman shifts between {raman_min} and {raman_max} found in: {file}"
                )

            selected_shifts = shifts[mask].astype(np.float32)

            if reference_shifts is None:
                reference_shifts = selected_shifts
            elif len(selected_shifts) != len(reference_shifts) or not np.allclose(
                selected_shifts, reference_shifts, rtol=0, atol=1e-4
            ):
                raise ValueError(f"Inconsistent Raman shifts found in: {file}")

            spectra = (
                data.iloc[1:, mask]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )

            if np.isnan(spectra).any():
                raise ValueError(f"Non-numeric or missing spectral values found in: {file}")

            if len(spectra):
                X.append(spectra)
                y.extend([label] * len(spectra))

    if not X:
        raise ValueError(f"No valid data found in: {root}")

    return (
        np.vstack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        reference_shifts,
    )


def augment_pet(X, y, ratio=AUGMENT_RATIO):
    pet = X[y == 1]
    n_new = max(0, int(np.sum(y == 0) * ratio) - len(pet))

    if n_new == 0:
        return X.copy(), y.copy()
    if len(pet) < 2:
        raise ValueError("At least two PET samples are required for interpolation.")

    i = np.random.randint(len(pet), size=n_new)
    j = np.random.randint(len(pet), size=n_new)
    same = i == j
    j[same] = (j[same] + 1) % len(pet)

    lam = np.random.rand(n_new, 1).astype(np.float32)
    synthetic = lam * pet[i] + (1 - lam) * pet[j]

    X_aug = np.vstack([X, synthetic]).astype(np.float32)
    y_aug = np.hstack([y, np.ones(n_new, dtype=np.int64)])
    order = np.random.permutation(len(y_aug))

    return X_aug[order], y_aug[order]


def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


class MultiScaleCNN(nn.Module):
    def __init__(self, kernels=KERNELS):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
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


def fit(X_train, y_train, X_val, y_val):
    X_aug, y_aug = augment_pet(X_train, y_train)
    train_loader = make_loader(X_aug, y_aug, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model = MultiScaleCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE).unsqueeze(1)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE).unsqueeze(1)
                y_batch = y_batch.to(DEVICE)

                loss = loss_fn(model(X_batch), y_batch)
                total_val_loss += loss.item() * X_batch.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best validation loss: {best_val_loss:.4f}"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_all(model, X):
    inputs = torch.from_numpy(X).float().to(DEVICE).unsqueeze(1)
    model.eval()

    with torch.no_grad():
        return torch.sigmoid(model(inputs)).cpu().numpy()


def calc_metrics(y_true, probability):
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Sensitivity": recall_score(y_true, prediction, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def plot_curves(y_true, probability, filename=CURVE_PATH):
    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    roc_auc = roc_auc_score(y_true, probability)
    pr_auc = average_precision_score(y_true, probability)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Random classifier")
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Sensitivity)",
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    baseline = np.mean(y_true)
    axes[1].plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Baseline = {baseline:.4f}",
    )
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall (Sensitivity)",
        ylabel="Precision",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, probability, filename=CM_PATH):
    prediction = (probability >= THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=["Background", "PET"],
    ).plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_metrics(metrics, filename=METRICS_PATH):
    names = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
    values = [metrics[name] for name in names]
    colors = ["#405D78", "#58758F", "#748FA7", "#9AABB9"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        names,
        values,
        color=colors,
        edgecolor="#334A5E",
        linewidth=0.8,
    )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    set_seed()

    for path in [MODEL_PATH, CURVE_PATH, CM_PATH, METRICS_PATH]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    X, y, raman_shifts = load_data(ROOT)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=SEED,
    )

    validation_ratio_within_train_val = VALIDATION_RATIO / (1.0 - TEST_RATIO)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio_within_train_val,
        stratify=y_train_val,
        random_state=SEED,
    )

    print(
        f"Train samples: {len(X_train)} | "
        f"Validation samples: {len(X_val)} | "
        f"Test samples: {len(X_test)}"
    )

    final_model = fit(X_train, y_train, X_val, y_val)
    test_probability = predict_all(final_model, X_test)
    metrics = calc_metrics(y_test, test_probability)

    plot_curves(y_test, test_probability)
    plot_confusion_matrix(y_test, test_probability)
    plot_metrics(metrics)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "kernels": KERNELS,
            "input_length": X.shape[1],
            "raman_min": RAMAN_MIN,
            "raman_max": RAMAN_MAX,
            "raman_shifts": raman_shifts,
            "epochs": EPOCHS,
            "augment_ratio": AUGMENT_RATIO,
            "threshold": THRESHOLD,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "test_metrics": metrics,
        },
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()