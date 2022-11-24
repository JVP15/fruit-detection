import cv2
import argparse
import mimetypes

from modules import preprocessor

from deepfruitvision import DeepFruitVision
from modules.view import View

DEFAULT_SOURCE = None
DEFAULT_DETECTION_WEIGHTS = 'weights/detection/best_classification.pt'
DEFAULT_RIPENESS_WEIGHTS = 'weights/ripeness/ripeness_model_fine_tuned_en2'
DEFAULT_DEFECT_WEIGHTS = 'weights/defect/defect_model_fine_tuned_en2'
DEFAULT_MIN_BOUNDING_BOX_SIZE = 0.1
DEFAULT_GUI = False
DEFAULT_DISPLAY = 'confidence'


def run(source = DEFAULT_SOURCE,
        detection_weights= DEFAULT_DETECTION_WEIGHTS,
        ripeness_weights = DEFAULT_RIPENESS_WEIGHTS,
        defect_weights = DEFAULT_DEFECT_WEIGHTS,
        min_bounding_box_size = DEFAULT_MIN_BOUNDING_BOX_SIZE,
        use_gui = DEFAULT_GUI,
        **kwargs):

    if source is None and not use_gui:
        raise ValueError('No source specified. Please either specify a source or use the GUI.')

    cap = None

    if source is not None:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError('Unable to open source: ' + str(source))

    preprocessor.MIN_BOUNDING_BOX_SIZE = min_bounding_box_size

    view = View()
    if use_gui:
        view.start()

    # init the pretrained networks
    deep_fruit_vision = DeepFruitVision(detection_weights, ripeness_weights, defect_weights, min_bounding_box_size)

    if use_gui:
        view.update_source(source)

    frame = None
    bounding_boxes = None

    while True: # to exit this loop, we just call `break`. It's not the most elegant solution, but this is a short script, so it works

        # this part of the program handles reading the frame and making predictions
        if source is not None:
            # if the source is an image, we only need to read and predict using it once
            if mimetypes.guess_type(source)[0].startswith('image'):
                frame = cv2.imread(source)

                if bounding_boxes is None:
                    bounding_boxes = deep_fruit_vision.get_harvestability(frame)

                # if we're not using the GUI, we can break out of the loop since we only do one prediction, otherwise, we stay in the loop to update the GUI
                if not use_gui:
                    break
            else:
                ret, frame = cap.read()

                if not ret and use_gui:
                    source = None
                    view.update_source(source)
                    continue
                # if there are no more frames, and we're not using the GUI, then quit (use break so that deep_fruit_vision.predict() does not get called)
                elif not ret:
                    break

                bounding_boxes = deep_fruit_vision.get_harvestability(frame)

        # this part is exlusive to the GUI
        if use_gui:
            display = view.get_display()

            if source is not None:
                display_frame = preprocessor.prepare_output_frame(frame, bounding_boxes, ui=display)
                view.update_image(display_frame)

            events, values = view.get_event()
            view.process_events(events, values)

            if view.quit: # if the user has quit, then exit the loop
                break

            # if the user has selected a new source, we need to update the video capture
            if source != view.source:

                # try to open the source from the GUI, but don't overwrite the video capture until we know it succeeds
                cap2 = cv2.VideoCapture(view.source)

                if cap2 is not None and cap2.isOpened():
                    if cap is not None and cap.isOpened(): # close the old video capture if it was open
                        cap.release()
                    cap = cap2
                    source = view.source
                    bounding_boxes = None
                else:
                    view.display_source_error(view.source)
                    view.update_source(source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Path to video file or camera index')
    parser.add_argument('--detection-weights', default=DEFAULT_DETECTION_WEIGHTS, help='Path to detection weights')
    parser.add_argument('--ripeness-weights', default=DEFAULT_RIPENESS_WEIGHTS, help='Path to ripeness weights')
    parser.add_argument('--disease-weights', default=DEFAULT_DEFECT_WEIGHTS, help='Path to disease weights')
    parser.add_argument('--min-bounding-box-size', default=DEFAULT_MIN_BOUNDING_BOX_SIZE, help='Minimum size of a bounding box before it is checked for ripeness and diseases')
    parser.add_argument('--use_gui', action='store_true', help='whether to use the gui or not')
    args = parser.parse_args()

    run(**vars(args))
