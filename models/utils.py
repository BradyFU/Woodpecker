import cv2
import supervision as sv
import numpy as np
from typing import List
import re
from PIL import Image

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area

    return iou

def extract_boxes(text):
    pattern = r'\[\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?),\s*([0-1](?:\.\d+)?)\s*\]'
    matches = re.findall(pattern, text)
    boxes = [list(map(float, match)) for match in matches]
    unique_boxes = set(tuple(box) for box in boxes)
    return [list(box) for box in unique_boxes]

def find_matching_boxes(extracted_boxes, entity_dict):
    phrases = []
    boxes = []
    for entity, info in entity_dict.items():
        for box in info['bbox']:
            if box in extracted_boxes:
                phrases.append(entity)
                boxes.append(box)
    return boxes, phrases

def annotate(image_path: str, boxes: List[List[float]], phrases: List[str]) -> np.ndarray:
    image_source = Image.open(image_path).convert("RGB")
    image_source = np.asarray(image_source)
    h, w, _ = image_source.shape
    if len(boxes) == 0:
        boxes = np.empty((0, 4))
    else:
        boxes = np.asarray(boxes) * np.array([w, h, w, h])
    
    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase}"
        for phrase
        in phrases
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


