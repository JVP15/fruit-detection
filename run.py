import cv2
import argparse
import torch
import sys

import preprocessor

from detection import FruitDetectionModule

DEFAULT_SOURCE = 0
DEFAULT_DETECTION_WEIGHTS = 'weights/detection/best.pt'
DEFAULT_RIPENESS_WEIGHTS = 'weights/ripeness/best.pt'
DEFAULT_DISEASE_WEIGHTS = 'weights/disease/best.pt'
DEFAULT_MIN_BOUNDING_BOX_SIZE = 0.15


def run(source = DEFAULT_SOURCE,
        detection_weights= DEFAULT_DETECTION_WEIGHTS,
        ripeness_weights = DEFAULT_RIPENESS_WEIGHTS,
        disease_weights = DEFAULT_DISEASE_WEIGHTS,
        min_bounding_box_size = DEFAULT_MIN_BOUNDING_BOX_SIZE,
        **kwargs):

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 3:
        detection_gpu = 'cuda:0'
        ripeness_gpu = 'cuda:1'
        disease_gpu = 'cuda:2'
    elif num_gpus == 2:
        detection_gpu = 'cuda:0'
        ripeness_gpu = 'cuda:1'
        disease_gpu = 'cuda:1'
    else:
        detection_gpu = 'cuda:0'
        ripeness_gpu = 'cuda:0'
        disease_gpu = 'cuda:0'

    preprocessor.MIN_BOUNDING_BOX_SIZE = min_bounding_box_size

    detection = FruitDetectionModule(detection_weights, device=detection_gpu)
    # ripeness = Ripeness(ripeness_weights, device=ripeness_gpu)
    # disease = Disease(disease_weights, device=disease_gpu)

    # if the source is an int, then it points to a camera, otherwise, it points to a video file
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f'Could not open video source {source}', file=sys.stderr)
        exit(1)

    ret, frame = cap.read()

    detection_input = preprocessor.preprocess_frame_for_detection(frame)
    bounding_boxes = detection.get_bounding_boxes(detection_input)

    while cap.isOpened() and ret:
        # localized_fruit = preprocessor.localize_fruit(frame, bounding_boxes)

        # ripenesses = preprocessor.get_ripeness(frame, localized_fruit)
        # diseases = preprocessor.get_disease(frame, localized_fruit)

        # preprocessor.prepare_output(frame, bounding_boxes, ripeness, diseases)

        ret, frame = cap.read()

        detection_input = preprocessor.preprocess_frame_for_detection(frame)
        bounding_boxes = detection.get_bounding_boxes(detection_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Path to video file or camera index')
    parser.add_argument('--detection-weights', default=DEFAULT_DETECTION_WEIGHTS, help='Path to detection weights')
    parser.add_argument('--ripeness-weights', default=DEFAULT_RIPENESS_WEIGHTS, help='Path to ripeness weights')
    parser.add_argument('--disease-weights', default=DEFAULT_DISEASE_WEIGHTS, help='Path to disease weights')
    parser.add_argument('--min-bounding-box-size', default=DEFAULT_MIN_BOUNDING_BOX_SIZE, help='Minimum size of a bounding box before it is checked for ripeness and diseases')
    args = parser.parse_args()

    run(**vars(args))
