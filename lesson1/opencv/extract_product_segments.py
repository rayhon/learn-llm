"""
extract_product_segments.py  •  v2
----------------------------------
Detect product segments in a short-form “home-gadgets” video, capture one
thumbnail per product, OCR the on-screen caption, deduplicate by title, and
export CSV + JSON.

Dependencies
------------
pip install opencv-python pillow pytesseract numpy pandas
(Plus the Tesseract binary: `brew install tesseract` or
 `sudo apt-get install tesseract-ocr`)
"""
from __future__ import annotations
import cv2, sys, json, re
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd

# ───────────────────────────── Scene detection ────────────────────────────── #
def detect_scene_changes(video_path: str,
                         sample_rate: float = 0.5,
                         threshold: float = 0.60) -> List[float]:
    """
    Return timestamps (seconds) where a fresh scene begins.
    *sample_rate* – seconds between histogram samples
    *threshold*   – Bhattacharyya distance that counts as a cut
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = int(max(1, fps * sample_rate))
    prev_hist, starts, frame_idx = None, [0.0], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                dist = cv2.compareHist(prev_hist.astype("float32"),
                                       hist.astype("float32"),
                                       cv2.HISTCMP_BHATTACHARYYA)
                # need ≥0.8 s between starts to avoid duplicates
                if dist >= threshold and (frame_idx / fps - starts[-1]) > 0.8:
                    starts.append(frame_idx / fps)
            prev_hist = hist
        frame_idx += 1
    cap.release()
    return starts

# ───────────────────────────── OCR helpers ────────────────────────────────── #
_label_re = re.compile(
    r"\b(\d{1,2})\s*[\.\-]?\s*([A-Za-z][A-Za-z0-9'’\- ]+)", re.UNICODE
)
_allowed  = re.compile(r"[^A-Za-z0-9'’\- ]+")

def clean_label(raw: str) -> str:
    """
    Extract and normalise "<rank>. <product>" from raw OCR text.
    Returns '' if no match.
    """
    m = _label_re.search(raw)
    if not m:
        return ""
    rank, label = m.groups()
    label = _allowed.sub("", label).strip()
    return f"{int(rank)}. {label}".lower()

def ocr_product_title(frame: np.ndarray) -> str:
    """Crop top 25 % (where TikTok puts captions) and OCR it."""
    h, w = frame.shape[:2]
    roi  = frame[0 : int(h * 0.25), :]
    raw  = pytesseract.image_to_string(Image.fromarray(roi), config="--psm 6")
    return clean_label(raw)

# ───────────────────────────── Main routine ───────────────────────────────── #
def main(video_path: str, out_dir: str = "product_segments"):
    Path(out_dir).mkdir(exist_ok=True)
    thumbs_dir = Path(out_dir, "thumbs")
    thumbs_dir.mkdir(exist_ok=True)

    starts = detect_scene_changes(video_path)
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)

    raw_records: List[Dict] = []

    for i, start in enumerate(starts):
        # grab a frame ~1 s after the cut for clarity
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start + 1) * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        # OCR label
        title = ocr_product_title(frame)
        if not title:
            continue  # skip segments with no valid caption

        # save thumbnail (300 px wide)
        thumb_path = thumbs_dir / f"seg_{i+1:02}.jpg"
        thumb = cv2.resize(frame, (300, int(frame.shape[0] * 300 / frame.shape[1])))
        cv2.imwrite(str(thumb_path), thumb)

        raw_records.append(
            dict(segment=i + 1,
                 start_time=round(start, 3),
                 product_title=title,
                 thumbnail=str(thumb_path))
        )

    cap.release()

    # ── Deduplicate by cleaned product_title ─────────────────────────────── #
    dedup: Dict[str, Dict] = {}
    for rec in raw_records:
        key = rec["product_title"]
        if key not in dedup or rec["start_time"] < dedup[key]["start_time"]:
            dedup[key] = rec                    # keep earliest occurrence

    records = sorted(dedup.values(), key=lambda r: r["start_time"])

    # ── Save outputs ─────────────────────────────────────────────────────── #
    with open(Path(out_dir, "products.json"), "w") as f:
        json.dump(records, f, indent=2)
    pd.DataFrame(records).to_csv(Path(out_dir, "products.csv"), index=False)

    print(f"✅ {len(records)} unique products saved to “{out_dir}/”")

# ───────────────────────────── Entrypoint ─────────────────────────────────── #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_product_segments.py YOUR_VIDEO.mp4")
        sys.exit(1)
    main(sys.argv[1])
