import cv2
import numpy as np
import streamlit as st
from tracker import Tracker
from ultralytics import YOLO
import pandas as pd
import time
import logging
import sys
import os

black_color = (0, 0, 0)  # Black color for text
white_color = (255, 255, 255)
yellow_color = (0, 255, 255)  # Yellow color for background
red_color = (0, 0, 255)  # Red color for lines
blue_color = (255, 0, 0)  # Blue color for lines
green_color = (0, 255, 0)

# Function to detect road area and calculate line coordinates
def detect_road_and_lines(frame):
    # Placeholder function for road detection
    # This function will be replaced with actual road detection code
    try:
        # Define region of interest (ROI) for road detection
        roi = frame[300:500, :]  # Adjust these values based on the position of the road in your video

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
        road_center_y_full_frame = road_center_y + 300  # Adding the offset used for ROI

        return road_center_y_full_frame
    except Exception as e:
        return None

# Function to detect vehicles in the frame
def detect_vehicles(frame, yolo_model, class_list, tracker):
    results = yolo_model.predict(frame)
    bbox_list = []

    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = result.tolist()
        class_name = class_list[int(class_id)]
        
        if class_name == 'car':
            bbox_list.append([int(x1), int(y1), int(x2), int(y2)])
    
    bbox_id = tracker.update(bbox_list)
    return bbox_id

def estimate_speed(bbox_id, frame, vh_down, vh_up, counter, counter1, line1_y, line2_y):
    detected_frames = []
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        bbox_height = y4 - y3
        offset = int(bbox_height * 0.2)

        if line1_y - offset < cy < line1_y + offset:
            vh_down[id] = time.time()
        if id in vh_down:
            if line2_y - offset < cy < line2_y + offset:
                elapsed_time = time.time() - vh_down[id]
                if id not in counter:
                    counter.append(id)
                    distance = 10  # meters
                    speed_mps = distance / elapsed_time
                    speed_kph = speed_mps * 3.6
                    cv2.putText(frame, f"Vehicle {id} going down at {speed_kph:.2f} km/h", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_frames.append((frame, id))
                    cv2.rectangle(frame, (x3, y3), (x4, y4), green_color, 2)
                    cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, yellow_color, 1)
                    cv2.putText(frame, str(int(speed_kph)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                yellow_color, 2)

        if line2_y - offset < cy < line2_y + offset:
            vh_up[id] = time.time()
        if id in vh_up:
            if line1_y - offset < cy < line1_y + offset:
                elapsed_time = time.time() - vh_up[id]
                if id not in counter1:
                    counter1.append(id)
                    distance = 10  # meters
                    speed_mps = distance / elapsed_time
                    speed_kph1 = speed_mps * 3.6
                    cv2.putText(frame, f"Vehicle {id} going up at {speed_kph1:.2f} km/h", (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_frames.append((frame, id))
                    cv2.rectangle(frame, (x3, y3), (x4, y4), green_color, 2)
                    cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, yellow_color, 1)
                    cv2.putText(frame, str(int(speed_kph1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                yellow_color, 2)

    return detected_frames

def main():
    st.title("Vehicle Speed Detection")
    st.write("Upload a video file to detect vehicle speed.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        video = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.video(file_bytes)

        if st.button("Estimate Speed"):
            try:
                video_path = 'temp.mp4'

                # Write the video data to a temporary file
                with open(video_path, 'wb') as f:
                    f.write(file_bytes)

                # Decode the numpy array as a video
                cap = cv2.VideoCapture(video_path)
                
                # Check if the video capture was successful
                if not cap.isOpened():
                    st.error("Error: Unable to open video file.")
                    return
                
                st.success("Video file opened successfully.")

                # Model - Yolo Version 8s
                model = YOLO('yolov8s.pt')

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
                counter=[]
                vh_up = {}
                counter1 = []
                frames_with_bboxes = []

                # Write frames with bounding boxes and speed information to a video file
                output_video_path = 'output_video.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, 30, (1020, 500))

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

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

                    # Detect vehicles in the frame
                    bbox_id = detect_vehicles(frame, model, class_list, tracker)

                    # Calculate speeds of detected vehicles
                    estimate_speed(bbox_id, frame, vh_down, vh_up, counter, counter1, line1_y, line2_y)

                    d = (len(counter))
                    u = (len(counter1))

                    cv2.putText(frame, ('Going Down : ') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, yellow_color, 2)
                    cv2.putText(frame, ('Going Up : ') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, yellow_color, 2)

                    # Write the frame to the output video
                    out.write(frame)

                    # Display the frame with bounding boxes and speed information
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                out.release()
                cap.release()
                cv2.destroyAllWindows()

                # Save the output video to a specific location
                output_video_dir = 'output_videos'
                os.makedirs(output_video_dir, exist_ok=True)
                output_video_filename = os.path.join(output_video_dir, 'output_video.mp4')
                os.rename(output_video_path, output_video_filename)

                return output_video_filename

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    output_video_path = main()
    if output_video_path:
        st.success("Video processing completed.")
        st.download_button(
            label="Download Output Video",
            data=open(output_video_path, "rb").read(),
            file_name="output_video.mp4",
            mime="video/mp4",
        )
