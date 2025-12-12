
# Builds a small clean face dataset from LFW (15 identities)
# Output: data/faces_clean/<person>/<img>.png

import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

import torch
from facenet_pytorch import MTCNN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "faces_clean"

# small dataset for running purposes
N_IDENTITIES = 15
MIN_FACES_PER_PERSON = 20
MAX_IMAGES_PER_ID = 30
IMG_SIZE = 160  # FaceNet standard

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("# downloading LFW (one-time) ...")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, resize=0.5)

    X = lfw.images.astype("float32")  # (n, h, w) grayscale
    y = lfw.target
    names = lfw.target_names

    # pick 15 random identities
    uniq_ids = np.unique(y)
    chosen = np.random.choice(uniq_ids, size=N_IDENTITIES, replace=False)

    # CPU usage is more reliable
    device = "cpu"

    mtcnn = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=False, device=device)

    total_saved = 0

    for pid in chosen:
        person = names[pid].replace(" ", "_")
        person_dir = OUT_DIR / person
        person_dir.mkdir(parents=True, exist_ok=True)

        idxs = np.where(y == pid)[0].tolist()
        random.shuffle(idxs)
        idxs = idxs[:MAX_IMAGES_PER_ID]

        saved = 0
        for idx in tqdm(idxs, desc=f"cropping {person}", leave=False):
            gray = X[idx]
            img = Image.fromarray((gray * 255).astype("uint8")).convert("RGB")

            face = mtcnn(img)
            if face is None:
                continue

            # face is torch tensor (3,160,160). save as png
            face_img = (face.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            out_path = person_dir / f"{person}_{saved:03d}.png"
            Image.fromarray(face_img).save(out_path)
            saved += 1

        total_saved += saved
        print(f"{person}: saved {saved} faces")

    print("\nDone.")
    print(f"Total faces saved: {total_saved}")
    print(f"Output folder: {OUT_DIR}")

if __name__ == "__main__":
    main()
