import argparse
from pynput.mouse import Listener
import json
from PIL import Image
import time
import pyautogui

parser = argparse.ArgumentParser(
    description="Record mouse movements and clicks."
)
# parser.add_argument("--retina", default=False, action=argparse.BooleanOptionalAction)

class Recorder:
    def __init__(self, scale, nodes):
        self.clicks = []
        self.scale = scale
        self.nodes = nodes
        self.sort_nodes_by_asc_area()
        self.listener = Listener(on_move=self.on_move, on_click=self.on_click)

    def sort_nodes_by_asc_area(self):
        self.nodes = sorted(
            (n for n in self.nodes if n["kind"] == "CONTROL"), 
            key=lambda n:
            (n["x2"] - n["x1"]) * (n["y2"] - n["y1"]))

    def find_node(self, x_screen, y_screen):
        """Find the node for which the bounding box contains x,y"""
        x_dom = int(x_screen * self.scale)
        y_dom = int(y_screen * self.scale)
        for n in self.nodes:
            if n["x1"] <= x_dom <= n["x2"] and n["y1"] <= y_dom <= n["y2"]:
                # Expects first hit to have the smallest area (sorted)
                return n
        return None

    def on_move(self, x, y):
        print(f"Mouse moved to ({x}, {y})")

    def on_click(self, x_screen, y_screen, button, pressed):
        """Only record ßon press on V1"""
        if pressed:
            node = self.find_node(x_screen, y_screen)
            if node:
                print(f"Found node: {node}")
                x_dom = int(x_screen * self.scale)
                y_dom = int(y_screen * self.scale)
                self.clicks.append(
                    {"node": node,
                    "x_screen": x_screen,
                    "y_screen": y_screen,
                    "x_dom": x_dom,
                    "y_dom": y_dom,
                    "action": "click",
                    }
                )
    
    def save_trace(self):
        with open('record.json', 'w') as f:
            json.dump(self.clicks, f, indent=4)

def main():
    args = parser.parse_args()
    dom = None
    screen_w, screen_h = pyautogui.size()
    with open('dom/dom.json') as f:
        dom = json.load(f)
    if dom is None:
        raise Exception("dom.json not loaded")    
    if dom["nodes"] is None or len(dom["nodes"]) == 0:
        raise Exception("No nodes found in dom.json")
    scale = dom["width"] / screen_w
    recorder = Recorder(scale, dom["nodes"])
    # Start the listener non-blockingly
    recorder.listener.start()
    try:
        # Periodically join with a timeout so Windows can process SIGINT (Ctrl+C)
        while recorder.listener.is_alive():
            recorder.listener.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        recorder.listener.stop()
        recorder.save_trace()

if __name__ == "__main__":
    main()