import cv2
import numpy as np
from yolo26mlx import YOLO

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Stable per-class BGR palette (same color for the same class every frame).
# Swap `cid` for `i` in `palette_index` below to get per-instance colors instead.
PALETTE = np.random.default_rng(seed=0).integers(64, 255, size=(80, 3), dtype=np.uint8).tolist()

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2


def draw_detections(frame, result):
    """Draw boxes + labels directly on a BGR frame using per-class colors."""
    if result.boxes is None or len(result.boxes) == 0:
        return
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i]]
        cid = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        name = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else f"class{cid}"
        palette_index = cid  # change to `i` for per-instance colors
        color = tuple(int(c) for c in PALETTE[palette_index % len(PALETTE)])
        label = f"{name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        ty = max(0, y1 - th - bl - 4)
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 6, ty + th + bl + 4), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, ty + th + 2),
            FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA,
        )


model = YOLO("models/yolo26n.npz")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25)
    draw_detections(frame, results[0])

    cv2.imshow("YOLO26 MLX", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
