from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detector = YOLO('./pretrained_models/detector.pt')
detector = detector.to(device)

DEMO_CONF_THRESHOLD = 0.3  # demo.py에서 실제로 쓰는 컷오프
LOW_CONF_THRESHOLD = 0.01  # 잘려나간 후보들까지 보기 위한 낮은 컷오프

img_paths = sorted(Path('glove_images').glob('*.jpg'))

print(f"{'image':<15} {'#pass(>=0.3)':<14} {'raw detections (class:conf)'}")
print("-" * 90)
for img_path in img_paths:
    img_cv2 = cv2.imread(str(img_path))
    # 낮은 threshold로 원시 검출 전부 확인
    raw_dets = detector(img_cv2, conf=LOW_CONF_THRESHOLD, verbose=False)[0]

    entries = []
    n_pass = 0
    for det in raw_dets:
        cls = det.boxes.cls.cpu().numpy().item()
        conf = det.boxes.conf.cpu().numpy().item()
        label = 'right' if cls == 1 else 'left'
        entries.append(f"{label}:{conf:.3f}")
        if conf >= DEMO_CONF_THRESHOLD:
            n_pass += 1

    entries_sorted = sorted(entries, key=lambda s: -float(s.split(':')[1]))
    print(f"{img_path.name:<15} {n_pass:<14} {', '.join(entries_sorted) if entries_sorted else '(no detections at all)'}")
