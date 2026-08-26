import glob, os, copy, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = "processed/preprocessed_particle_spectra"
MODEL_PATH = "models/PE_PET_lipid.pth"
CM_PATH = "processed/PE_PET_lipid_Confusion_Matrix.png"
METRICS_PATH = "processed/PE_PET_lipid_Classification_Metrics.png"
HISTORY_PATH = "processed/PE_PET_lipid_Training_History.png"

SEED, EPOCHS, BATCH_SIZE, LR = 42, 50, 8, 1e-4
VALIDATION_RATIO, TEST_RATIO = 0.1, 0.2
PATIENCE, MIN_DELTA = 5, 1e-3
CLASS_NAMES = ["PE", "PET", "lipid"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12,
    "axes.labelsize": 12, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11
})

def set_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(root):
    X, y, ref = [], [], None
    for cls, label in {"PE": 0, "PET": 1, "lipid": 2}.items():
        path = os.path.join(root, cls)
        if not os.path.isdir(path): raise FileNotFoundError(f"Class folder not found: {path}")
        for file in sorted(glob.glob(os.path.join(path, "*.csv"))):
            df = pd.read_csv(file, header=None)
            shifts = pd.to_numeric(df.iloc[0], errors="coerce").to_numpy()
            mask = np.isfinite(shifts)
            shifts = shifts[mask].astype(np.float32)

            if ref is None: ref = shifts
            elif len(shifts) != len(ref) or not np.allclose(shifts, ref, atol=1e-4):
                raise ValueError(f"Inconsistent Raman shifts found in: {file}")

            spectra = df.iloc[1:, mask].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
            if np.isnan(spectra).any(): raise ValueError(f"Invalid values found in: {file}")
            if len(spectra):
                X.append(spectra); y.extend([label] * len(spectra))

    if not X: raise ValueError(f"No valid data found in: {root}")
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int64), ref

def make_loader(X, y=None, shuffle=False):
    tensors = [torch.from_numpy(X).float()]
    if y is not None: tensors.append(torch.from_numpy(y).long())
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle)

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
            nn.Linear(64 * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def fit(X_train, y_train, X_val, y_val):
    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val)
    model = CNNAdaptivePool2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_state, patience = float("inf"), None, 0
    h = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        model.train(); loss_sum = correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE).unsqueeze(1), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb); loss = loss_fn(logits, yb)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        train_loss = loss_sum / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval(); loss_sum = correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE).unsqueeze(1), yb.to(DEVICE)
                logits = model(xb)
                loss_sum += loss_fn(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()

        val_loss = loss_sum / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        for k, v in zip(h, [train_loss, val_loss, train_acc, val_acc]): h[k].append(v)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}"
        )

        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None: model.load_state_dict(best_state)
    return model, h

def predict(model, X):
    probs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in make_loader(X):
            probs.append(torch.softmax(model(xb.to(DEVICE).unsqueeze(1)), 1).cpu().numpy())
    return np.vstack(probs)

def macro_metrics(y, prob):
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    values = []
    for i in range(3):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        values.append([
            (tp + tn) / cm.sum(),
            tp / (tp + fn) if tp + fn else 0,
            tn / (tn + fp) if tn + fp else 0,
            tp / (tp + fp) if tp + fp else 0
        ])
    return np.mean(values, axis=0), pred

def plot_metrics(values):
    names = ["Accuracy", "Sensitivity", "Specificity", "Precision"]
    colors = ["#405D78", "#58758F", "#748FA7", "#9AABB9"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="#334A5E", linewidth=0.8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(METRICS_PATH, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def plot_confusion(y, pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y, pred, labels=[0, 1, 2]),
        display_labels=CLASS_NAMES
    ).plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
    ax.set_title("")
    fig.tight_layout(); fig.savefig(CM_PATH, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def plot_history(h):
    e = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    ax[0].plot(e, h["train_accuracy"], linewidth=2, label="Train")
    ax[0].plot(e, h["val_accuracy"], linewidth=2, label="Validation")
    ax[0].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    ax[0].grid(alpha=0.25); ax[0].legend()

    ax[1].plot(e, h["train_loss"], linewidth=2, label="Train")
    ax[1].plot(e, h["val_loss"], linewidth=2, label="Validation")
    ax[1].set(xlabel="Epoch", ylabel="Loss")
    ax[1].grid(alpha=0.25); ax[1].legend()

    fig.tight_layout(); fig.savefig(HISTORY_PATH, dpi=300, bbox_inches="tight")
    plt.show(); plt.close(fig)

def main():
    set_seed()
    os.makedirs("models", exist_ok=True); os.makedirs("processed", exist_ok=True)
    X, y, shifts = load_data(ROOT)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VALIDATION_RATIO / (1 - TEST_RATIO),
        stratify=y_tv, random_state=SEED
    )

    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    print("Class mapping: PE = 0, PET = 1, lipid = 2")

    model, history = fit(X_train, y_train, X_val, y_val)
    prob = predict(model, X_test)
    metrics, pred = macro_metrics(y_test, prob)

    plot_history(history)
    plot_confusion(y_test, pred)
    plot_metrics(metrics)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "class_mapping": {"PE": 0, "PET": 1, "lipid": 2},
        "input_length": X.shape[1],
        "raman_shifts": shifts,
        "architecture": "3-layer CNN + AdaptiveAvgPool1d(2) + 2 hidden Dense layers",
        "pool_output_size": 2,
        "conv_channels": [16, 32, 64],
        "kernel_sizes": [9, 6, 3],
        "conv_strides": [4, 3, 1],
        "maxpool_kernel": 2,
        "dense_features": [128, 64, 32, 3],
        "dropout": 0.3,
        "weight_decay": 1e-3,
        "macro_metrics": dict(zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision"], metrics
        )),
        "history": history
    }, MODEL_PATH)

if __name__ == "__main__":
    main()