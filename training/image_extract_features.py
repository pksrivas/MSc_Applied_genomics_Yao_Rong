import os, random, torch, numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

DATASET_DIR = "dataset/images"
OUTPUT_FEATURES, OUTPUT_LABELS = "processed/DINO_features.npy", "processed/DINO_labels.npy"
IMG_SIZE, BATCH_SIZE, SEED = 224, 32, 42
CLASS_TO_IDX = {"lipid": 0, "plastic": 1}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((.485, .456, .406), (.229, .224, .225))
])

paths, labels = [], []
for cls, label in CLASS_TO_IDX.items():
    folder = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(folder): raise FileNotFoundError(f"Class folder not found: {folder}")
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            paths.append(os.path.join(folder, f)); labels.append(label)

features = []
with torch.no_grad():
    for i in tqdm(range(0, len(paths), BATCH_SIZE)):
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+BATCH_SIZE]]).to(device)
        features.append(model(x).cpu().numpy())

os.makedirs("processed", exist_ok=True)
np.save(OUTPUT_FEATURES, np.concatenate(features))
np.save(OUTPUT_LABELS, np.asarray(labels, dtype=np.int64))