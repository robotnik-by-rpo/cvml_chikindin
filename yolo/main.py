import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from torchvision import transforms

root = Path(__file__).parent
model_path = root / r"runs\detect\figures\yolo\weights\best.pt"
model = YOLO(model_path)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ret, frame = camera.read()
    annotated = frame.copy()
    key = cv2.waitKey(10) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if key == ord("q"):
        break
    results = model(image)
    for result in results:
        if result.boxes is not None:
            cls = result.boxes.cls
            if len(cls) > 0:
                boxes_xyxy = result.boxes.xyxy
                confidences = result.boxes.conf
                cls = result.boxes.cls
                cls_names = [result.names[int(c)] for c in cls]
                for  (box, conf, name) in zip(boxes_xyxy, confidences, cls_names):
                    x1, y1, x2, y2 = box.tolist()
                    color = (0, 0, 255) if name == "cube" else (0, 255, 0)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, f"{name}: {conf:.2f}", (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)   
    cv2.imshow("Camera", annotated)
        