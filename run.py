import time

import cv2
import argparse
import torch
import sys

from modules import preprocessor

from modules.detection import FruitDetectionModule
from modules.disease import DiseaseModule
from modules.ripeness import RipenessModule
from modules.view import View

DEFAULT_SOURCE = None
DEFAULT_DETECTION_WEIGHTS = 'weights/detection/best.pt'
DEFAULT_RIPENESS_WEIGHTS = 'weights/ripeness/mobilenetv2'
DEFAULT_DISEASE_WEIGHTS = 'weights/disease/resnet'
DEFAULT_MIN_BOUNDING_BOX_SIZE = 0.1
DEFAULT_GUI = False
DEFAULT_DISPLAY = 'confidence'

def run(source = DEFAULT_SOURCE,
        detection_weights= DEFAULT_DETECTION_WEIGHTS,
        ripeness_weights = DEFAULT_RIPENESS_WEIGHTS,
        disease_weights = DEFAULT_DISEASE_WEIGHTS,
        min_bounding_box_size = DEFAULT_MIN_BOUNDING_BOX_SIZE,
        use_gui = DEFAULT_GUI,
        **kwargs):


    if source is None and not use_gui:
        raise ValueError('No source specified. Please either specify a source or use the GUI.')

    if source is not None:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError('Unable to open source: ' + str(source))

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

    view = View()
    if use_gui:
        view.start()

    # init the pretrained networks
    detection_module = FruitDetectionModule(detection_weights, device=detection_gpu)
    ripeness_module = RipenessModule(ripeness_weights, device=ripeness_gpu)
    disease_module = DiseaseModule(disease_weights, device=disease_gpu)

    if use_gui:
        view.update_source(source)

    # if we are using the GUI, we need to check if the user has quit.
    # If we are not using the GUI, view.quit is always false, so it will loop until the video is over.
    while not view.quit:

        if source is not None:
            ret, frame = cap.read()
            if not ret and use_gui:
                source = None
                view.update_source(source)
                continue
            elif not ret:
                break

            detection_input = preprocessor.preprocess_frame_for_detection(frame)
            bounding_boxes = detection_module.get_bounding_boxes(detection_input)

            localized_fruit = preprocessor.localize_fruit(frame, bounding_boxes)

            ripenesses = ripeness_module.get_ripeness_predictions(localized_fruit)
            diseases = disease_module.get_disease_predictions(localized_fruit)

        if use_gui:
            display = view.get_display()

            if source is not None:
                display_frame = preprocessor.prepare_output_frame(frame, bounding_boxes, ripenesses, diseases, ui=display)
                view.update_image(display_frame)

            events, values = view.get_event()
            view.process_events(events, values)

            # if the user has selected a new source, we need to update the video capture
            if source != view.source:

                # try to open the source from the GUI, but don't overwrite the video capture until we know it succeeds
                cap2 = cv2.VideoCapture(view.source)

                if cap2 is not None and cap2.isOpened():
                    if cap.isOpened(): # close the old video capture if it was open
                        cap.release()
                    cap = cap2
                    source = view.source
                else:
                    view.display_source_error(view.source)
                    view.update_source(source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Path to video file or camera index')
    parser.add_argument('--detection-weights', default=DEFAULT_DETECTION_WEIGHTS, help='Path to detection weights')
    parser.add_argument('--ripeness-weights', default=DEFAULT_RIPENESS_WEIGHTS, help='Path to ripeness weights')
    parser.add_argument('--disease-weights', default=DEFAULT_DISEASE_WEIGHTS, help='Path to disease weights')
    parser.add_argument('--min-bounding-box-size', default=DEFAULT_MIN_BOUNDING_BOX_SIZE, help='Minimum size of a bounding box before it is checked for ripeness and diseases')
    parser.add_argument('--use_gui', action='store_true', help='whether to use the gui or not')
    args = parser.parse_args()

    run(**vars(args))
