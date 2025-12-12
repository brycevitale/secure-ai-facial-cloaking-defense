# Builds "edited" dataset from clean (JPEG + blur)
# Output: data/faces_edited/<person>/<same_name>.png

from pathlib import Path
from PIL import Image, ImageFilter
import io

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "faces_clean"
EDITED = ROOT / "data" / "faces_edited"

def jpeg_reencode(img: Image.Image, quality: int = 25) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def blur(img: Image.Image, radius: float = 1.5) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def main():
    if not CLEAN.exists():
        print("Missing clean folder:", CLEAN)
        print("Run: python src/prepare_lfw.py")
        return

    EDITED.mkdir(parents=True, exist_ok=True)

    for person_dir in CLEAN.iterdir():
        if not person_dir.is_dir():
            continue

        out_person = EDITED / person_dir.name
        out_person.mkdir(parents=True, exist_ok=True)

        for img_path in person_dir.glob("*.png"):
            img = Image.open(img_path).convert("RGB")

            # simple "everyday edits"
            img2 = jpeg_reencode(img, quality=25)
            img2 = blur(img2, radius=1.5)

            out_path = out_person / img_path.name
            img2.save(out_path)

    print("Edited faces saved to:", EDITED)

if __name__ == "__main__":
    main()


