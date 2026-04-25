import torch
import torchvision
from pathlib import Path
import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np
import datetime


PUSHUP = 0
last_person_time = None
was_down = False

def get_angle(a,b,c):
    cb = np.atan2(c[1]-b[1], c[0]-b[0])
    ab = np.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def counting_push_up(annotated, keypoints):
    global was_down, PUSHUP
    h, w = annotated.shape[:2]
    
    def is_valid_point(x, y):
        return x > 50 and x < w - 50 and y > 50 and y < h - 50
    

    left_hand_visible = (
        keypoints[5][0] > 0 and is_valid_point(keypoints[5][0], keypoints[5][1]) and
        keypoints[7][0] > 50 and is_valid_point(keypoints[7][0], keypoints[7][1]) and 
        keypoints[9][0] > 50 and is_valid_point(keypoints[9][0], keypoints[9][1])  
    )
    
    right_hand_visible = (
        keypoints[6][0] > 0 and is_valid_point(keypoints[6][0], keypoints[6][1]) and
        keypoints[8][0] > 50 and is_valid_point(keypoints[8][0], keypoints[8][1]) and
        keypoints[10][0] > 50 and is_valid_point(keypoints[10][0], keypoints[10][1])
    )
    
    if left_hand_visible and right_hand_visible:
        left_angle = get_angle(keypoints[5], keypoints[7], keypoints[9])
        right_angle = get_angle(keypoints[6], keypoints[8], keypoints[10])
        is_down = ((left_angle + right_angle) / 2 ) < 140
        with open("log.log","a", encoding="utf-8") as f:
            f.write(f"l {left_angle}, r {right_angle}, {datetime.datetime.now()},{PUSHUP}\n")
        print("l", left_angle)
        print("r", right_angle)
        if is_down:
            was_down = True
            return False
        
        elif was_down and not is_down:
            was_down = False
            return True
        else:
            return False

    else:
        was_down = False
        return False

root = Path(__file__).parent
model_path = root / "yolo26n-pose.pt"

model = YOLO("yolo26n-pose.pt")

camera = cv2.VideoCapture(0)
ps = None
while camera.isOpened():
    
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    t = time.perf_counter()
    results = model(frame)
    print(f"FPS {(1/(time.perf_counter()-t)):.1f}")

    if not results:
        continue


    result = results[0]
    keypoints = result.keypoints.xy.tolist()

    if not keypoints:
        if last_person_time is None:
            last_person_time = time.time()
        
        if time.time() - last_person_time > 4:
            PUSHUP = 0
            ps = ps.stop()
            last_person_time = None
            was_down = False
        cv2.putText(frame, f"Push up: {PUSHUP}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
        cv2.imshow("Pose",frame)
        continue

    last_person_time = time.time()

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    if counting_push_up(annotated=annotated,keypoints=keypoints[0]):
        PUSHUP += 1 
        if ps is None:
            ps = playsound("Survivor - Eye Of The Tiger (Album Version).mp3", block = False)
    cv2.putText(annotated, f"Push up: {PUSHUP}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
    cv2.imshow("Pose",annotated)

