import time

import cv2
import argparse
import torch
import sys

import preprocessor

from detection import FruitDetectionModule
from disease import DiseaseModule
from ripeness import RipenessModule

DEFAULT_SOURCE = 0
DEFAULT_DETECTION_WEIGHTS = 'weights/detection/best.pt'
DEFAULT_RIPENESS_WEIGHTS = 'weights/ripeness/mobilenetv2'
DEFAULT_DISEASE_WEIGHTS = 'weights/disease/resnet'
DEFAULT_MIN_BOUNDING_BOX_SIZE = 0.15


def run(source = DEFAULT_SOURCE,
        detection_weights= DEFAULT_DETECTION_WEIGHTS,
        ripeness_weights = DEFAULT_RIPENESS_WEIGHTS,
        disease_weights = DEFAULT_DISEASE_WEIGHTS,
        min_bounding_box_size = DEFAULT_MIN_BOUNDING_BOX_SIZE,
        **kwargs):

    num_gpus = torch.cuda.device_count()

    detection_gpu = 'cuda:0'
    ripeness_gpu = '/GPU:0'
    disease_gpu = '/GPU:0'

    if num_gpus >= 2:
        ripeness_gpu = '/GPU:1'
        disease_gpu = '/GPU:1'
    if num_gpus >= 3:
        disease_gpu = '/GPU:2'

    preprocessor.MIN_BOUNDING_BOX_SIZE = min_bounding_box_size

    detection_module = FruitDetectionModule(detection_weights, device=detection_gpu)
    ripeness_module = RipenessModule(ripeness_weights, device=ripeness_gpu)
    disease_module = DiseaseModule(disease_weights, device=disease_gpu)

    # if the source is an int, then it points to a camera, otherwise, it points to a video file
    #cap = cv2.VideoCapture(source)
    cap = cv2.VideoCapture('dataset/images/test/image_%01d.png', cv2.CAP_IMAGES)

    if not cap.isOpened():
        print(f'Could not open video source {source}', file=sys.stderr)
        exit(1)

    ret, frame = cap.read()

    detection_input = preprocessor.preprocess_frame_for_detection(frame)
    bounding_boxes = detection_module.get_bounding_boxes(detection_input)

    while cap.isOpened():

        localized_fruit = preprocessor.localize_fruit(frame, bounding_boxes)

        ripenesses = ripeness_module.get_ripeness_predictions(localized_fruit)
        diseases = disease_module.get_disease_predictions(localized_fruit)

        display_frame = preprocessor.prepare_output_frame(frame, bounding_boxes, ripenesses, diseases)

        ret, frame = cap.read()

        if not ret:
            break

        detection_input = preprocessor.preprocess_frame_for_detection(frame)
        bounding_boxes = detection_module.get_bounding_boxes(detection_input)

        cv2.imshow('image', display_frame)
        cv2.waitKey(30)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Path to video file or camera index')
    parser.add_argument('--detection-weights', default=DEFAULT_DETECTION_WEIGHTS, help='Path to detection weights')
    parser.add_argument('--ripeness-weights', default=DEFAULT_RIPENESS_WEIGHTS, help='Path to ripeness weights')
    parser.add_argument('--disease-weights', default=DEFAULT_DISEASE_WEIGHTS, help='Path to disease weights')
    parser.add_argument('--min-bounding-box-size', default=DEFAULT_MIN_BOUNDING_BOX_SIZE, help='Minimum size of a bounding box before it is checked for ripeness and diseases')
    args = parser.parse_args()

    run(**vars(args))
