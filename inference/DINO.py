import os, torch, joblib, numpy as np
from PIL import Image
import torchvision.transforms as T

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "training", "models", "DINO_SVM.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_CLASS = {0: "lipid", 1: "plastic"}
IMG_SIZE = 224

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
).to(DEVICE)

model.eval()
svm_model = joblib.load(MODEL_PATH)

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


def extract_feature(image):
    x = transform(
        Image.fromarray(image).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        return model(x).cpu().numpy()


def predict_image(image):
    try:
        feature = extract_feature(image)

        if hasattr(svm_model, "predict_proba"):
            probs = svm_model.predict_proba(feature)[0]
            pos = int(np.argmax(probs))
            pred = int(svm_model.classes_[pos])
            prob = float(probs[pos])
        else:
            pred = int(svm_model.predict(feature)[0])
            prob = 1.0

        if pred not in IDX_TO_CLASS:
            raise ValueError(
                f"Unexpected SVM class index: {pred}"
            )

        return IDX_TO_CLASS[pred], prob

    except Exception as e:
        print("DINO inference error:", e)
        return "Error", 0.0