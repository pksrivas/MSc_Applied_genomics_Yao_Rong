import os, joblib, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

FEATURE_PATH = "processed/DINO_features.npy"
LABEL_PATH = "processed/DINO_labels.npy"
MODEL_DIR, OUTPUT_DIR = "models", "processed"
SEED, TEST_RATIO = 42, 0.2

plt.rcParams.update({
    "font.family":"Arial","font.size":12,"axes.titlesize":14,
    "axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11
})

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return {
        "Accuracy": accuracy_score(y,p),
        "Sensitivity": recall_score(y,p,zero_division=0),
        "Specificity": tn/(tn+fp) if tn+fp else 0,
        "Precision": precision_score(y,p,zero_division=0)
    }

def plot_metrics(m, path):
    names = ["Accuracy","Sensitivity","Specificity","Precision"]
    vals = [m[n] for n in names]
    colors = ["#405D78","#58758F","#748FA7","#9AABB9"]
    fig, ax = plt.subplots(figsize=(7,5))
    bars = ax.bar(names, vals, color=colors, edgecolor="#334A5E", linewidth=0.8)
    ax.set_ylim(0,1.08); ax.set_ylabel("Score"); ax.grid(axis="y",alpha=0.25); ax.set_axisbelow(True)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)

def plot_cm(y, p, path):
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(confusion_matrix(y,p,labels=[0,1]), display_labels=["Lipid","Plastic"]).plot(
        ax=ax, values_format="d", colorbar=False, cmap="Blues"
    )
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)

def evaluate(model, name, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    p = model.predict(X_test)
    joblib.dump(model, os.path.join(MODEL_DIR,f"DINO_{name}.pkl"))
    plot_metrics(metrics(y_test,p), os.path.join(OUTPUT_DIR,f"DINO_{name}_Classification_Metrics.png"))
    plot_cm(y_test,p, os.path.join(OUTPUT_DIR,f"DINO_{name}_Confusion_Matrix.png"))

def main():
    os.makedirs(MODEL_DIR,exist_ok=True); os.makedirs(OUTPUT_DIR,exist_ok=True)
    X, y = np.load(FEATURE_PATH), np.load(LABEL_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=TEST_RATIO,random_state=SEED,stratify=y
    )

    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1),
        "SVM": Pipeline([
            ("scaler",StandardScaler()),
            ("svm",SVC(kernel="rbf",probability=True,random_state=SEED))
        ])
    }

    for name, model in models.items():
        evaluate(model,name,X_train,X_test,y_train,y_test)

if __name__ == "__main__":
    main()