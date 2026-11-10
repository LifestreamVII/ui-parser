from detector import Detector

# Config
IMAGE_PATH = "input.jpg"
OUTPUT_PATH = "output.jpg"
# Inference model path
MODEL_PATH = "weights/omniparser/icon_detect_v3/model.pt"
# Model hyperparameters
IMGSZ = 1280
STRIDES = (8, 16, 32)
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.45

def main() -> None:
    detector = Detector(MODEL_PATH, IMGSZ, STRIDES, CONF_THRESHOLD, IOU_THRESHOLD)
    detector.load_image(IMAGE_PATH)
    detector.detect_objects()
    detector.draw_detections(OUTPUT_PATH)

if __name__ == "__main__":
    main()
