import cv2
import argparse
import torch
import sys

import preprocessor

from detection import FruitDetectionModule
from disease import DiseaseModule

DEFAULT_SOURCE = 0
DEFAULT_DETECTION_WEIGHTS = 'weights/detection/best.pt'
DEFAULT_RIPENESS_WEIGHTS = 'weights/ripeness/best.pt'
DEFAULT_DISEASE_WEIGHTS = 'weights/disease/resnet'
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
        disease_gpu = '/GPU:2'
    elif num_gpus == 2:
        detection_gpu = 'cuda:0'
        ripeness_gpu = 'cuda:1'
        disease_gpu = '/GPU:1'
    else:
        detection_gpu = 'cuda:0'
        ripeness_gpu = 'cuda:0'
        disease_gpu = '/GPU:0'

    preprocessor.MIN_BOUNDING_BOX_SIZE = min_bounding_box_size

    detection_module = FruitDetectionModule(detection_weights, device=detection_gpu)
    # ripeness = Ripeness(ripeness_weights, device=ripeness_gpu)
    disease_module = DiseaseModule(disease_weights, device=disease_gpu)

    # if the source is an int, then it points to a camera, otherwise, it points to a video file
    cap = cv2.VideoCapture(source)
    #cap = cv2.VideoCapture('dataset/images/test/apple_%01d.jpg', cv2.CAP_IMAGES)

    if not cap.isOpened():
        print(f'Could not open video source {source}', file=sys.stderr)
        exit(1)

    ret, frame = cap.read()

    detection_input = preprocessor.preprocess_frame_for_detection(frame)
    bounding_boxes = detection_module.get_bounding_boxes(detection_input)

    while cap.isOpened():
        localized_fruit = preprocessor.localize_fruit(frame, bounding_boxes)

        # ripenesses = preprocessor.get_ripeness(frame, localized_fruit)
        diseases = disease_module.get_disease_predictions(localized_fruit)

        # preprocessor.prepare_output(frame, bounding_boxes, ripeness, diseases)
        for bounding_box, disease_pred in zip(bounding_boxes, diseases):
            print('Class', bounding_box['class'], 'Confidence', bounding_box['conf'], 'Disease', disease_pred)
        print()

        ret, frame = cap.read()

        if not ret:
            break

        detection_input = preprocessor.preprocess_frame_for_detection(frame)
        bounding_boxes = detection_module.get_bounding_boxes(detection_input)

    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Path to video file or camera index')
    parser.add_argument('--detection-weights', default=DEFAULT_DETECTION_WEIGHTS, help='Path to detection weights')
    parser.add_argument('--ripeness-weights', default=DEFAULT_RIPENESS_WEIGHTS, help='Path to ripeness weights')
    parser.add_argument('--disease-weights', default=DEFAULT_DISEASE_WEIGHTS, help='Path to disease weights')
    parser.add_argument('--min-bounding-box-size', default=DEFAULT_MIN_BOUNDING_BOX_SIZE, help='Minimum size of a bounding box before it is checked for ripeness and diseases')
    args = parser.parse_args()

    run(**vars(args))
