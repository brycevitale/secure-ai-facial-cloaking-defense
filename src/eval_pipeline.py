# evaluation
# FaceNet embeddings + 1-NN identification across conditions

import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import torch
from facenet_pytorch import InceptionResnetV1

IMG_SIZE = 160

ROOT = Path(__file__).resolve().parents[1]           # repo/
PROJECT = ROOT.parent                                # secure_ai_full/
DATA = PROJECT / "data"

CLEAN = DATA / "faces_clean"
EDITED = DATA / "faces_edited"
CLOAKED = DATA / "faces_cloaked"
CLOAKED_JPEG = DATA / "faces_cloaked_jpeg"
CLOAKED_BLUR = DATA / "faces_cloaked_blur"


def load_img(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def list_identity_items(root_dir: Path):
    items = []
    for person_dir in sorted(root_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for p in sorted(person_dir.glob("*.png")):
            items.append((p, label))
    return items


def make_split(clean_dir: Path, train_frac: float = 0.7):
    # per identity split so every person appears in train/test
    by_id = {}
    for p, lab in list_identity_items(clean_dir):
        by_id.setdefault(lab, []).append(p)

    train, test = [], []
    for lab, ps in by_id.items():
        cut = max(1, int(len(ps) * train_frac))
        for p in ps[:cut]:
            train.append((p, lab))
        for p in ps[cut:]:
            test.append((p, lab))
    return train, test


def embed(model, items, device):
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for p, lab in tqdm(items, desc="embedding", leave=False):
            arr = load_img(p)
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            e = model(x).cpu().numpy()[0]
            embs.append(e)
            labels.append(lab)

    embs = np.vstack(embs).astype("float32")
    embs = normalize(embs) 
    return embs, labels


def top1_accuracy(train_embs, train_labels, test_embs, test_labels):
    clf = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    clf.fit(train_embs, train_labels)
    pred = clf.predict(test_embs)
    pred = np.array(pred)
    test = np.array(test_labels)
    return float((pred == test).mean())


def map_test_items(test_list, new_root: Path):
    # assumes same person folders and same filenames
    mapped = []
    for p, lab in test_list:
        mapped_path = new_root / lab / p.name
        mapped.append((mapped_path, lab))
    return mapped


def check_paths(items, label):
    missing = [str(p) for p, _ in items if not p.exists()]
    if missing:
        print(f"[!] Missing files for {label}: {len(missing)}")
        print("Example missing:", missing[0])
        return False
    return True


def main():
    for d in [CLEAN, EDITED, CLOAKED, CLOAKED_JPEG, CLOAKED_BLUR]:
        if not d.exists():
            print("Missing folder:", d)
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InceptionResnetV1(pretrained="vggface2").to(device)

    train_list, test_list = make_split(CLEAN, train_frac=0.7)
    print("Train:", len(train_list))
    print("Test:", len(test_list))

    t0 = time.time()
    train_embs, train_labels = embed(model, train_list, device)
    clean_embs, clean_labels = embed(model, test_list, device)
    clean_acc = top1_accuracy(train_embs, train_labels, clean_embs, clean_labels)
    print(f"Clean accuracy: {clean_acc*100:.2f}%")

    results = {"clean": clean_acc}

    conditions = [
        ("edited", EDITED),
        ("cloaked", CLOAKED),
        ("cloaked_jpeg", CLOAKED_JPEG),
        ("cloaked_blur", CLOAKED_BLUR),
    ]

    for name, folder in conditions:
        mapped = map_test_items(test_list, folder)
        if not check_paths(mapped, name):
            return
        embs, labels = embed(model, mapped, device)
        acc = top1_accuracy(train_embs, train_labels, embs, labels)
        results[name] = acc
        print(f"{name} accuracy: {acc*100:.2f}%")

    print(f"\nTotal eval time: {time.time()-t0:.1f}s")
    print("\nRESULTS")
    for k in ["clean", "edited", "cloaked", "cloaked_jpeg", "cloaked_blur"]:
        print(f"{k}: {results[k]*100:.2f}%")

if __name__ == "__main__":
    main()
