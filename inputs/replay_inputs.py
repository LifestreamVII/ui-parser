import argparse
from pynput.mouse import Button, Controller
import json
import cv2
import sys
import time
import numpy as np
import pyautogui

parser = argparse.ArgumentParser(
    description="Replay a recording sequence."
)
# parser.add_argument("--retina", default=False, action=argparse.BooleanOptionalAction)

class Player:
    def __init__(self, sequence, dom_t, dom_s, student_img, scale):
        self.sequence = sequence
        self.dom_t = dom_t
        self.dom_s = dom_s
        self.scale = scale
        self.student_img = student_img
        self.nodes_map = {}
        self.mouse = Controller()
        self.associate_nodes()

    def load_student_image(image):
        self.student_img = image

    def norm(self, x: float, width: int) -> float:
        """x as a fraction of screen width: 0.0 = left edge, 1.0 = right edge."""
        return x / width

    def pct_diff(self, a: float, b: float) -> float:
        """Symmetric % difference (0 when identical, immune to which side is bigger)."""
        ref = max(abs(a), abs(b))
        return 0.0 if ref == 0 else abs(a - b) / ref * 100

    def close(self, cx1, cy1, w1, h1, cx2, cy2, w2, h2, tol=3.0):
        return (self.pct_diff(cx1 / w1, cx2 / w2) < tol
            and self.pct_diff(cy1 / h1, cy2 / h2) < tol)

    def _legacy_template_match(self, crop1, crop2):
        """Establish a match between two cropped captures to compare nodes"""
        img1 = cv2.imread("dom_t/" + crop1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread("dom/" + crop2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        res = cv2.absdiff(img1, img2)
        res = res.astype(np.uint8)

        return 100 - (np.count_nonzero(res) * 100) / res.size


    def _legacy_associate_nodes(self):
        """Associate nodes between teach/stud setups"""
        t_width, t_height = self.dom_t["width"], self.dom_t["height"]
        s_width, s_height = pyautogui.size().width * self.scale, pyautogui.size().height * self.scale
        for node_t in self.dom_t["nodes"]:
            candidates = [] # [{id, score}]
            for node_s in self.dom_s["nodes"]:
                score = 0
                if node_t["kind"] != node_s["kind"]:
                    continue
                if "label" in node_t and "label" in node_s and node_t["label"] == node_s["label"]:
                    score += 50
                if (self.close(
                    node_t["cx"], 
                    node_t["cy"], 
                    t_width,
                    t_height,
                    node_s["cx"], 
                    node_s["cy"],
                    s_width,
                    s_height
                )):
                    # Close enough to be considered the same node
                    score += 10
                if "crop_path" in node_t and "crop_path" in node_s:
                    t_match = self.template_match(node_t["crop_path"], node_s["crop_path"])
                    score += t_match
                if score > 0:
                    candidates.append({"id": node_s["id"], "score": score, "cx": node_s["cx"], "cy": node_s["cy"]})
            candidates.sort(key=lambda c: c["score"], reverse=True)
            if len(candidates) > 0:
                self.nodes_map[node_t["id"]] = candidates[0]

    def associate_nodes(self):
        img = cv2.imread(self.student_img)
        grayscale_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for node in self.dom_t["nodes"]:
            if node["kind"] != "CONTROL":
                continue
            crop = cv2.imread(f"dom_t/{node['crop_path']}")
            if crop is None:
                continue
            best = None
            candidates = []
            for s in [0.5, 1.0, 1.5, 2.0]:
                # crop is at teacher scale; student may be 2x
                # resize to student scale
                t = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
                grayscale_crop = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(grayscale_s, grayscale_crop, cv2.TM_CCOEFF_NORMED)
                # mx: match value, lx, ly: location, h, w: size
                _, mx, _, loc = cv2.minMaxLoc(res)
                if best is None or mx > best[0]:
                    best = (mx, loc, t.shape, s)
                if mx > 0.6:
                    candidates.append((mx, loc, t.shape, s))
            if best and best[0] > 0.6: # 0.6 is a threshold
                if len(candidates) > 1:
                    candidates.sort(key=lambda x: self.close(
                        x[1][0] + x[2][1] / 2,
                        x[1][1] + x[2][0] / 2,
                        self.dom_s["width"],
                        self.dom_s["height"],
                        node["cx"] * x[3],
                        node["cy"] * x[3],
                        self.dom_t["width"] * x[3],
                        self.dom_t["height"] * x[3]
                    ), reverse=True)
                    best = candidates[0]
                mx, (lx, ly), (h, w, _), _ = best
                self.nodes_map[node["id"]] = {"score": mx, "cx": lx + w / 2, "cy": ly + h / 2}

    def play_step(self, step=0):
        action = self.sequence[step]["action"]
        target_og = self.sequence[step]["node"]["id"]
        if target_og not in self.nodes_map:
            print(f"Target {target_og} not found in nodes_map")
            return
        target = self.nodes_map[target_og]
        # assume click
        cx, cy = target["cx"] / self.scale, target["cy"] / self.scale
        self.mouse.position = (cx, cy)
        time.sleep(0.2)
        print(f"{target_og} (TEACH) matched with {target['id']} (STUD) at final screen coordinates {cx}, {cy}")
        self.mouse.click(Button.left, 1)

def main():
    args = parser.parse_args()
    dom_t = None
    dom_s = None
    sequence = None
    screen_w, screen_h = pyautogui.size()
    student_img = "student_rearranged.png"
    with open('dom_t/dom.json') as f:
        dom_t = json.load(f)
    with open('dom/dom_s.json') as f:
        dom_s = json.load(f)
    with open('record.json') as f:
        sequence = json.load(f)
    if dom_s is None or dom_t is None:
        raise Exception("Missing dom_s.json or dom_t.json")
    if sequence is None:
        raise Exception("Missing record.json")
    if dom_s["nodes"] is None or len(dom_s["nodes"]) == 0:
        raise Exception("No nodes found in dom_s.json")
    if dom_t["nodes"] is None or len(dom_t["nodes"]) == 0:
        raise Exception("No nodes found in dom_t.json")
    scale = dom_s["width"] / screen_w
    player = Player(sequence, dom_t, dom_s, student_img, scale)

    player.play_step(0)

if __name__ == "__main__":
    main()