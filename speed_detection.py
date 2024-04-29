import cv2
import numpy as np
import ultralytics
import pandas as pd
from ultralytics import YOLO
import time
from math import dist
from tracker import *

# Function to detect road area and calculate line coordinates
def detect_road_and_lines(frame):
    # Define region of interest (ROI) for road detection
    roi = frame[300:500, :] 

    # Apply lane detection algorithm to detect road area
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)

    # Check if any lines are detected
    if lines is None or len(lines) == 0:
        return None, None

    # Calculate center position of the road
    line1_y = np.min([lines[:, :, 1].min(), lines[:, :, 3].min()])
    line2_y = np.max([lines[:, :, 1].max(), lines[:, :, 3].max()])
    road_center_y = (line1_y + line2_y) // 4

    # Convert road center position to full frame coordinates
    road_center_y_full_frame = road_center_y + 300 

    return int(road_center_y_full_frame)

# Model - Yolo Version 8s
model = YOLO('yolov8s.pt')

# Video Input
cap = cv2.VideoCapture('veh2.mp4')

# Defining the Class list
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

tracker = Tracker()
count = 0

vh_down = {}
counter = []

vh_up = {}
counter1 = []

black_color = (0, 0, 0)  # Black color for text
white_color = (255, 255, 255)
yellow_color = (0, 255, 255)  # Yellow color for background
red_color = (0, 0, 255)  # Red color for lines
blue_color = (255, 0, 0)  # Blue color for lines
green_color = (0, 255, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Detect road area and calculate line coordinates
    road_center_y = detect_road_and_lines(frame)

    if road_center_y is not None:
        # Define the distance between the two lines (adjust as needed)
        line_distance = 60

        # Calculate the positions of the two lines
        line1_y = road_center_y - line_distance // 2
        line2_y = road_center_y + line_distance // 2

        # Draw lines with the calculated positions
        line_thickness = 2
        cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), red_color, line_thickness)
        cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), blue_color, line_thickness)

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
            #cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,white_color,1)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), red_color, 2)
            

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        
        bbox_height = y4 - y3  # Height of the bounding box
        offset = int(bbox_height * 0.2)
        cy1 = line1_y
        cy2 = line2_y

        # Going DOWN
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.rectangle(frame, (x3, y3), (x4, y4), green_color, 2)
                    cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, yellow_color, 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                yellow_color, 2)

        # Going UP
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.rectangle(frame, (x3, y3), (x4, y4), green_color, 2)
                    cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, yellow_color, 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                yellow_color, 2)

    d = (len(counter))
    u = (len(counter1))

    cv2.putText(frame, ('Going Down : ') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, yellow_color, 2)
    cv2.putText(frame, ('Going Up : ') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, yellow_color, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

