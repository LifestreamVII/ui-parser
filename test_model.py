from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms

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


def letterbox_image(image: Image.Image, target_size: int = IMGSZ) -> tuple[torch.Tensor, float]:
    """Resize image preserving aspect ratio and pad into a square canvas."""
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Pad with gray background (114, 114, 114) standard for YOLO
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(resized, (0, 0))

    # Convert to CHW float tensor normalized to [0, 1]
    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float()[None] / 255.0
    return tensor, scale


def decode_outputs(
    outputs: tuple[torch.Tensor, ...],
    scale: float,
    orig_w: int,
    orig_h: int,
    strides: tuple[int, ...] = STRIDES,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode raw TorchScript YOLOv9 heads into bounding boxes and confidence scores.
    Outputs tuple contains alternating (cls_logits, box_ltrb) tensors per stride.
    """
    boxes_list: list[torch.Tensor] = []
    scores_list: list[torch.Tensor] = []

    for i, stride in enumerate(strides):
        cls_logits = outputs[2 * i].sigmoid()[0, 0]
        ltrb = outputs[2 * i + 1][0]
        grid_h, grid_w = cls_logits.shape[-2], cls_logits.shape[-1]

        gy, gx = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32, device=device),
            torch.arange(grid_w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        cx, cy = gx + 0.5, gy + 0.5
        left, top, right, bottom = ltrb

        boxes_list.append(
            torch.stack(
                [
                    (cx - left) * stride,
                    (cy - top) * stride,
                    (cx + right) * stride,
                    (cy + bottom) * stride,
                ],
                dim=-1,
            ).reshape(-1, 4)
        )
        scores_list.append(cls_logits.reshape(-1))

    all_boxes = torch.cat(boxes_list, dim=0)
    all_scores = torch.cat(scores_list, dim=0)

    # Filter by confidence threshold
    valid_mask = all_scores > conf_threshold
    boxes = all_boxes[valid_mask]
    scores = all_scores[valid_mask]

    if len(boxes) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device)

    # Apply Non-Maximum Suppression (single-class detection)
    nms_indices = batched_nms(
        boxes,
        scores,
        torch.zeros_like(scores, dtype=torch.long),
        iou_threshold,
    )
    boxes = boxes[nms_indices]
    scores = scores[nms_indices]

    # Map boxes from letterboxed canvas back to original image dimensions
    boxes = boxes / scale
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)

    return boxes, scores


def draw_detections(
    image_path: str | Path,
    output_path: str | Path,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Draw bounding boxes and confidence labels on the original image."""
    image_cv = cv2.imread(str(image_path))
    if image_cv is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, np.round(box))
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        text_y = max(y1 - 4, 12)
        cv2.putText(
            image_cv,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image_cv)


def main() -> None:
    model_path = Path(MODEL_PATH)
    image_path = Path(IMAGE_PATH)

    if not model_path.exists():
        fallback = Path("model.pt")
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Ensure you have downloaded weights with dl_omniparser.py."
            )

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at '{IMAGE_PATH}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load TorchScript model
    print(f"Loading TorchScript model from {model_path}...")
    t0 = time.perf_counter()
    model = torch.jit.load(str(model_path), map_location=device).eval()
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Loaded image '{image_path}' ({orig_w}x{orig_h})")

    input_tensor, scale = letterbox_image(image, target_size=IMGSZ)
    input_tensor = input_tensor.to(device)

    # Run inference
    print("Running inference...")
    t1 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
    infer_time = time.perf_counter() - t1
    print(f"Inference completed in {infer_time * 1000:.1f}ms")

    # Decode predictions and apply NMS
    boxes, scores = decode_outputs(
        outputs=outputs,
        scale=scale,
        orig_w=orig_w,
        orig_h=orig_h,
        strides=STRIDES,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=device,
    )

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    print(f"Detected {len(boxes_np)} interactable elements (conf > {CONF_THRESHOLD})")

    # Draw and save bounding boxes
    draw_detections(image_path, OUTPUT_PATH, boxes_np, scores_np)
    print(f"Annotated image saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
