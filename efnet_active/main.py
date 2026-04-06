from train_model import predict
import time
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if key == ord("p"):
            t = time.perf_counter()
            label, confidence = predict(frame)
            print(time.perf_counter()-t)
            print(label, confidence)
        elif key == ord("q"):
            exit(1)