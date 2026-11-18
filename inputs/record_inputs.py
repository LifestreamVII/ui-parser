import argparse
from pynput.mouse import Listener
import json
from PIL import Image


parser = argparse.ArgumentParser(
    description="Record mouse movements and clicks."
)
parser.add_argument("--retina", default=False, action=argparse.BooleanOptionalAction)

class Recorder:
    def __init__(self, width, height, nodes, retina):
        self.clicks = []
        self.width = width
        self.height = height
        self.nodes = nodes
        self.retina = retina
        self.sort_nodes_by_asc_area()
        self.listener = Listener(on_move=self.on_move, on_click=self.on_click)

    def sort_nodes_by_asc_area(self):
        self.nodes = sorted(
            (n for n in self.nodes if n["kind"] == "CONTROL"), 
            key=lambda n:
            (n["x2"] - n["x1"]) * (n["y2"] - n["y1"]))

    def find_node(self, x, y):
        """Find the node for which the bounding box contains x,y"""
        if self.retina:
            screen_width = 1920
            factor = self.width / screen_width
            x = int(x * factor)
            y = int(y * factor)
        print(f"Click pressed at ({x}, {y})")
        for n in self.nodes:
            if n["x1"] <= x <= n["x2"] and n["y1"] <= y <= n["y2"]:
                # Expects first hit to have the smallest area (sorted)
                return n
        return None

    def on_move(self, x, y):
        print(f"Mouse moved to ({x}, {y})")

    def on_click(self, x, y, button, pressed):
        """Only record ßon press on V1"""
        if pressed:
            node = self.find_node(x, y)
            if node:
                print(f"Found node: {node}")
                self.clicks.append(
                    {"node": node, "x": x, "y": y, "action": "click"}
                )
    
    def save_trace(self):
        with open('record.json', 'w') as f:
            json.dump(self.clicks, f, indent=4)

def main():
    args = parser.parse_args()
    dom = None
    retina = False
    if args.retina:
        retina = True
        with open('dom/dom.json') as f:
            dom = json.load(f)
        if dom is None:
            raise Exception("dom.json not loaded")    
        if dom["nodes"] is None or len(dom["nodes"]) == 0:
            raise Exception("No nodes found in dom.json")
        recorder = Recorder(dom["width"], dom["height"], dom["nodes"], retina)
        try:
            with recorder.listener as listener:
                listener.join()
        except KeyboardInterrupt:
            print("Recording stopped by user")
        finally:
            recorder.save_trace()

if __name__ == "__main__":
    main()