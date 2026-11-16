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
    output_image = Path(args.out + "debug_annotated.png")
    output_json = Path(args.out + "dom.json")
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
            "cx": get_cx(n),
            "cy": get_cy(n),
            "score": n["score"],
            "kind": n["kind"]
        })
    ocr_nodes = ocr_text(args.image, args.ocr_conf)
    pairs = pair_nodes(Path(args.out) / "crops", ocr_nodes, nodes, args.image)
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
        "nodes": pairs + ocr_nodes,
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
                "cx": get_cx({
                       "x1": x1,
                       "x2": x2
                }),
                "cy": get_cy({
                    "y1": y1,
                    "y2": y2
                }),
                "score": score
            })
    return ocr_nodes

def pair_nodes(out_path, ocr_nodes, control_nodes, image):
    nodes = []
    for control in control_nodes:
        best_cx = None
        best = None
        control_w = control["x2"] - control["x1"]
        control_h = control["y2"] - control["y1"]
        best_w, best_h = None, None
        for ocr in ocr_nodes:
            if abs(control["cy"] - ocr["cy"]) < 20:
                # same row
                gap_x = min(
                    abs(control["x1"] - ocr["x2"]),
                    abs(ocr["x1"] - control["x2"])
                )
                if best_cx is None or gap_x < best_cx:
                    best_cx = gap_x
                    best = ocr
        if best:
            nodes.append({**control, "crop_path": f"crops/{control['id']}.png", "label": best["label"]})
        else:
            nodes.append({**control, "crop_path": f"crops/{control['id']}.png"})
            best = control
        best_w = best["x2"] - best["x1"]
        best_h = best["y2"] - best["y1"]
        union_box = (min(control["x1"], best["x1"]), min(control["y1"], best["y1"]),
            max(control["x2"], best["x2"]), max(control["y2"], best["y2"]))
        save_crops(
            out_path,
            image,
            control["id"],
            union_box
        )
    return nodes

def save_crops(out_path, image, id, region):
    img_input = Path(image)
    img = Image.open(img_input)
    W, H = img.size
    left, top, right, bottom = region
    region = (max(0, left - 8), max(0, top - 8),
        min(W, right + 8), min(H, bottom + 8))
    img = img.crop(region).save(Path(out_path) / f"{id}.png")

def get_cx(node):
    return node["x1"] + (node["x2"] - node["x1"]) / 2

def get_cy(node):
    return node["y1"] + (node["y2"] - node["y1"]) / 2

if __name__ == "__main__":
    main()