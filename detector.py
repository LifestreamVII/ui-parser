from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms


class Detector:
    def __init__(self, path, imgsz, strides, conf_threshold, iou_threshold):
         self._model = None
         self._device = None
         self.path = path
         self.imgsz = imgsz
         self.strides = strides
         self.conf_threshold = conf_threshold
         self.iou_threshold = iou_threshold
         self.image = None
         self.image_path = None
         self.size = None
         self.result = None
         self.detections = None
         self.load_model()

    def load_model(self) -> None:
        model_path = Path(self.path)

        if not model_path.exists():
            fallback = Path("model.pt")
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading model from {model_path} using device {self._device}")
            t0 = time.perf_counter() # Start timer
            self._model = torch.jit.load(model_path, map_location=self._device)
            self._model.eval() # Puts model in inference mode
            print(f"Model loaded in {time.perf_counter() - t0:.2f}s")
        except:
            raise Exception(f"Failed to load model from {model_path}")

    def load_image(self, image_path: str) -> None:
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found at {image_path}")
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise Exception(f"Unsupported image format: {image_path.suffix}")
            self.image = Image.open(image_path)
            self.size = self.image.size
            self.image = self.image.convert("RGB")
            self.image_path = image_path
        except:
            raise Exception(f"Failed to load image from {image_path}")

    def detect_objects(self) -> None:
        try:
            t0 = time.perf_counter() # Start timer
            width, height = self.size
            input_tensor, scale = self._letterbox(self.image, self.imgsz)
            input_tensor = input_tensor.to(self._device)
            # Run inference
            print ("Running inference...")
            with torch.no_grad():
                outputs = self._model(input_tensor.unsqueeze(0))
            infer_time = time.perf_counter() - t0
            print(f"Inference time: {infer_time:.4f} seconds")
            boxes, scores = self._decode_outputs(outputs, scale, width, height)
            print("Decoding outputs...")
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()
            print(f"Detected {len(boxes_np)} interactable elements (conf > {self.conf_threshold})")
            self.detections = [
                {
                    "kind": "CONTROL",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": s
                }
                for (x1, x2, y1, y2), s in zip(boxes_np, scores_np)
            ]
        except:
            raise Exception("Failed to run inference")

    def _letterbox(self, image: Image.Image, target_size: int = 0) -> tuple[torch.Tensor, float]:
        # Resizing ------------------------------------------------------
        width, height = self.size
        if not self.imgsz or target_size == 0:
            raise Exception("Letterbox: Target size not set")
        if target_size == 0:
            target_size = self.imgsz
        # Preserve aspect ratio, highest dimension is target_size
        scale = min(target_size / width, target_size / height)
        new_w, new_h = round(width * scale), round(height * scale)
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        # Padding -------------------------------------------------------
        bg_rgb = (114, 114, 114)  # Standard gray background for YOLO
        canvas = Image.new("RGB", (target_size, target_size), bg_rgb)
        canvas.paste(resized, (0, 0))
        # Tensor conversion ---------------------------------------------
        # Convert HWC to CHW float tensor, normalize to [0, 1]
        tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1)
        tensor = tensor.float() / 255.0
        return tensor, scale
    
    def _decode_outputs(
        self,
        outputs: tuple[torch.Tensor, ...],
        scale: float,
        orig_w: int,
        orig_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode raw TorchScript YOLOv9 heads into bounding boxes and confidence scores.
        Outputs tuple contains alternating (cls_logits, box_ltrb) tensors per stride.
        """
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []

        for i, stride in enumerate(self.strides):
            cls_logits = outputs[2 * i].sigmoid()[0, 0]
            ltrb = outputs[2 * i + 1][0]
            grid_h, grid_w = cls_logits.shape[-2], cls_logits.shape[-1]

            gy, gx = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.float32, device=self._device),
                torch.arange(grid_w, dtype=torch.float32, device=self._device),
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
        valid_mask = all_scores > self.conf_threshold
        boxes = all_boxes[valid_mask]
        scores = all_scores[valid_mask]

        if len(boxes) == 0:
            return torch.empty((0, 4), device=self._device), torch.empty((0,), device=self._device)

        # Apply Non-Maximum Suppression (single-class detection)
        nms_indices = batched_nms(
            boxes,
            scores,
            torch.zeros_like(scores, dtype=torch.long),
            self.iou_threshold,
        )
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]

        # Map boxes from letterboxed canvas back to original image dimensions
        boxes = boxes / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)

        return boxes, scores

    def draw_detections(self, output: str) -> None:
        if not self.detections:
            raise ValueError("No detections to draw.")
        output = Path(output)
        image_path = Path(self.image_path)
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            raise ValueError(f"Failed to read image from {image_path}")
        for x1, y1, x2, y2, score in map(lambda x: (
            int(np.round(x["x1"])),
            int(np.round(x["x2"])),
            int(np.round(x["y1"])),
            int(np.round(x["y2"])),
            x["score"]),
            self.detections):
            cv2.rectangle(
                image_cv,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            label = f"{score:.2f}"
            text_y = max(y1 - 4, 12)
            cv2.putText(image_cv, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(output, image_cv)
