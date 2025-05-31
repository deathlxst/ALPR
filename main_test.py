# 0. Setup Paths and Imports
import os
import sys
import cv2
import torch
import numpy as np
import pathlib
import easyocr
import time
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Any
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, Future
from sort.sort import Sort
from tkinter import Tk, filedialog
import ast
import re

OCRBox = List[Tuple[float, float]]
OCRResultType = Tuple[List[List[int]], str, float]

# Patch for Windows path if needed
pathlib.PosixPath = pathlib.WindowsPath

# === YOLOv5 setup ===
YOLOV5_PATH = r"C:\YOLO\my_license\yolov5"
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

device = select_device("0" if torch.cuda.is_available() else "cpu")
model_vehicle = DetectMultiBackend(os.path.join(YOLOV5_PATH, "yolov5s.pt"), device=device)
model_plate = DetectMultiBackend(os.path.join(YOLOV5_PATH, "my_licenses.pt"), device=device)
reader = easyocr.Reader(["en"])

# === Utility Functions ===
def validate_indonesian_plate(text: str) -> str:
    text = text.replace("-", "").replace(".", "").replace(",", "").replace(" ", "").upper()
    pattern = r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,4})$'
    match = re.match(pattern, text)
    if not match:
        return ""
    prefix, digits, suffix = match.groups()
    suffix = suffix[:3]  # Keep only first 3 characters
    return f"{prefix} {digits} {suffix}"

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # (height, width)
    if shape[0] == 0 or shape[1] == 0:
        return None  # Prevent invalid resize

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    if new_unpad[0] == 0 or new_unpad[1] == 0:
        return None  # Again, avoid resizing to 0 width/height

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    h_start = (new_shape[0] - new_unpad[1]) // 2
    w_start = (new_shape[1] - new_unpad[0]) // 2
    canvas[h_start:h_start + new_unpad[1], w_start:w_start + new_unpad[0]] = img_resized
    return canvas, ratio, (w_start, h_start)


def detect_objects(image, model, conf_thres=0.3, iou_thres=0.4):
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return []

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = letterbox(img_rgb, (640, 640))
    if result is None:
        return []
    img_resized, ratio, dwdh = result

    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.permute(2, 0, 1).contiguous().float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    results = model(img_tensor)
    pred = non_max_suppression(results, conf_thres, iou_thres)

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes((640, 640), det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2, y2, conf.item(), int(cls)))
    return detections


def filter_text(region: np.ndarray, ocr_result: List[OCRResultType], line_tolerance: int = 20,
                min_height_ratio: float = 0.25, confidence_threshold: float = 0.3) -> Tuple[List[str], List[OCRBox]]:
    if not ocr_result:
        return [], []
    region_h = region.shape[0]
    texts_with_geometry = []

    for result in ocr_result:
        if len(result) != 3:
            continue
        box, text, conf = result
        if conf < confidence_threshold:
            continue
        y_center = float(np.mean([pt[1] for pt in box]))
        x_center = float(np.mean([pt[0] for pt in box]))
        height = float(np.linalg.norm(np.array(box[0]) - np.array(box[3])))
        height_ratio = height / region_h
        if height_ratio >= min_height_ratio:
            texts_with_geometry.append((y_center, x_center, text, box))

    texts_with_geometry.sort(key=lambda x: x[0])
    if not texts_with_geometry:
        return [], []

    top_y = texts_with_geometry[0][0]
    filtered = [(x, text, box) for y, x, text, box in texts_with_geometry if abs(y - top_y) <= line_tolerance]
    filtered.sort(key=lambda x: x[0])
    texts = [f[1] for f in filtered]
    boxes = [f[2] for f in filtered]
    return texts, boxes

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return (xcar1, ycar1, xcar2, ycar2, car_id)
    return None

def ocr_text_only(plate_img) -> Tuple[str, float, np.ndarray]:
    try:
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_upscaled = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        ocr_result = reader.readtext(plate_upscaled, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        filtered_texts, boxes = filter_text(plate_img, ocr_result)
        raw_text = "".join(filtered_texts) if filtered_texts else ""
        final_text = validate_indonesian_plate(raw_text)
        print("Raw OCR Text:", raw_text)
        print("Final Plate:", final_text)
        conf_score = np.mean([r[2] for r in ocr_result]) if ocr_result else 0.0
        return final_text, conf_score, plate_upscaled
    except Exception as e:
        print("[OCR ERROR]", e)
        return "OCR Error", 0.0, plate_img

# === Persistent Tracking with Plate Memory ===
track_plates = {}  # track_id -> (text, conf, bbox_rel)

def update_plate_memory(track_id: int, text: str, conf: float, bbox_rel):
    if text and conf > 0.3:
        track_plates[track_id] = (text, conf, bbox_rel)

def get_plate_memory(track_id: int):
    return track_plates.get(track_id, ("", 0.0, None))
    

def select_video_file():
    root = Tk()
    root.withdraw()
    root.update()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    root.destroy()
    return video_path

# === Main Processing Function ===
def main():
    video_path = select_video_file()
    print("Running on video:", video_path)

    if not video_path or not os.path.exists(video_path):
        print("Error: Video file not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

    log_data = []
    prev_time = 0
    executor = ThreadPoolExecutor(max_workers=1)
    tracker = Sort()
    ocr_futures = {}
    ocr_results = {}
    last_detection_time = {}
    detection_cooldown = 5
    frame_nmr = -1

    while True:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        vehicle_dets = detect_objects(frame, model_vehicle)
        det_array = np.array([[*det[:4], det[4]] for det in vehicle_dets if det[5] in [2, 3, 5, 7]])
        tracks = tracker.update(det_array)

        best_conf = 0.0
        best_plate = ""

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if (track_id not in ocr_futures or
                (ocr_futures[track_id].done() and current_time - last_detection_time.get(track_id, 0) > detection_cooldown)):

                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size == 0:
                    continue

                plates = detect_objects(vehicle_roi, model_plate)

            for plate in plates:
                # Convert plate coords to absolute frame coords
                px1, py1, px2, py2, *_ = plate
                abs_plate = (x1 + px1, y1 + py1, x1 + px2, y1 + py2, plate[4], plate[5])

                vehicle_info = get_car(abs_plate, [(x1, y1, x2, y2, track_id)])
                if not vehicle_info:
                    continue  # Skip plates that don't belong to the current vehicle

                plate_img = frame[abs_plate[1]:abs_plate[3], abs_plate[0]:abs_plate[2]]
                if plate_img.size == 0:
                    continue  # Invalid crop

                plate_h, plate_w = plate_img.shape[:2]
                aspect_ratio = plate_w / plate_h
                if not (1.5 <= aspect_ratio <= 5):  # Optional: avoid strange boxes
                    continue

                ocr_futures[track_id] = executor.submit(ocr_text_only, plate_img)
                last_detection_time[track_id] = current_time

                cv2.rectangle(frame, (abs_plate[0], abs_plate[1]), (abs_plate[2], abs_plate[3]), (0, 255, 0), 2)
                break 

            if track_id in ocr_futures and ocr_futures[track_id].done():
                result, conf, crop_img = ocr_futures[track_id].result()
                if conf > best_conf:
                    best_conf = conf
                    best_plate = result
                ocr_results[track_id] = (result, crop_img)

            if track_id in ocr_results:
                text, crop = ocr_results[track_id]
                crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2))
                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                H, W = crop.shape[:2]
                top = max(0, y1 - H - 40)
                left = max(0, x1 + (x2 - x1 - W) // 2)
                # Get frame size
                frame_h, frame_w = frame.shape[:2]

                # Adjust width if overflow
                if left + W > frame_w:
                    W = frame_w - left
                    crop = crop[:, :W]

                # Adjust height if overflow
                if top + H > frame_h:
                    H = frame_h - top
                    crop = crop[:H, :]

                # Check again to avoid empty crop
                if W > 0 and H > 0:
                    frame[top:top+H, left:left+W] = crop

                # Ensure crop fits inside the frame
                frame_h, frame_w = frame.shape[:2]
                if top + H > frame_h:
                    H = frame_h - top
                    crop = crop[:H, :]
                if left + W > frame_w:
                    W = frame_w - left
                    crop = crop[:, :W]
                cv2.rectangle(frame, (left, top - 70), (left + W, top), (255, 255, 255), -1)
                cv2.putText(frame, text, (left + 10, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        if best_plate:
            log_data.append({"frame": frame_nmr, "plate": best_plate, "confidence": best_conf})

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        out.write(frame)
        cv2.imshow("ALPR from Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    executor.shutdown()

    if log_data:
        df_log = pd.DataFrame(log_data)
        df_log.to_csv("detection_log.csv", index=False)
        print(f"CSV log saved with {len(log_data)} entries.")
    else:
        print("No valid license plates detected; CSV not written.")

if __name__ == "__main__":
    main()