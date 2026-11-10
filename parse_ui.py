import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from detector import Detector
import json
import hashlib
import random

# Config
# Inference model path
MODEL_PATH = "weights/omniparser/icon_detect_v3/model.pt"
# Model hyperparameters
IMGSZ = 1280
STRIDES = (8, 16, 32)
IOU_THRESHOLD = 0.45

parser = argparse.ArgumentParser(
    description="Parse UI elements from an image."
)
parser.add_argument("--image", required=True)
# (Most) checkboxes scored at 0.08-0.15 confidence
parser.add_argument("--conf", type=float, default=0.05)
parser.add_argument("--ocr-conf", type=float, default=0.5)
parser.add_argument("--out", default="dom/")

# OCR engine
ocr = RapidOCR()

def main():
    args = parser.parse_args()
    output_image = args.out + "/debug_annotated.png"
    output_json = args.out + "dom.json"
    detector = Detector(MODEL_PATH, IMGSZ, STRIDES, args.conf, IOU_THRESHOLD)
    detector.load_image(args.image)
    control_detections = detector.detect_objects()
    w, h = detector.size
    nodes = []
    for n in control_detections:
        if (n["x2"] - n["x1"]) <= 4 or (n["y2"] - n["y1"]) <= 4:
            continue
        hash_object = hashlib.sha256(str('%d%d%d%d%d' % (n["x1"], n["y1"], n["x2"], n["y2"], n["score"])).encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        node_id = hash_hex[:9]
        nodes.append({
            "id": node_id,
            "x1": n["x1"],
            "x2": n["x2"],
            "y1": n["y1"],
            "y2": n["y2"],
            "score": n["score"],
            "kind": n["kind"]
        })
    ocr_nodes = ocr_text(args.image, args.ocr_conf)
    detections = nodes + ocr_nodes
    detector.draw_detections(
        detections,
        args.image,
        output_image
    )
    obj = {
        "image": args.image,
        "width": w,
        "height": h,
        "nodes": detections,
    }
    j = json.dumps(obj, indent=4)
    with open(output_json, "w") as f:
        f.write(j)
    print(f"Output image saved to {output_image} and JSON saved to {output_json}")

def ocr_text(image_path, conf: float) -> list[dict]:
    img = cv2.imread(image_path)
    results, _ = ocr(img)
    ocr_nodes = []
    if results:
        for box, text, score in results:
            if score < conf:
                continue
            # RapidOCR returns a 4-point bounding box
            # [[x1,y1], [x2,y1], [x3,y3], [x4,y4]]
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            hash_object = hashlib.sha256(str('%d%d%d%d%d' % (x1, y1, x2, y2, score)).encode('utf-8'))
            hash_hex = hash_object.hexdigest()
            node_id = hash_hex[:9]
            ocr_nodes.append({
                "kind": "TEXT", "label": text, "id": node_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": score
            })
    return ocr_nodes


if __name__ == "__main__":
    main()