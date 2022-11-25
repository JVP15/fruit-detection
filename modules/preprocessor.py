import cv2
import numpy as np
import tensorflow as tf
from typing import List

from modules import ripeness, defect

LOCALIZED_IMAGE_SIZE = (224, 224)

def square_bounding_box(bounding_box):
    """Makes a bounding box square, modifying the original bounding box.
    Assumes that the values are normalized between 0 and 1."""

    xmin = bounding_box['xmin']
    ymin = bounding_box['ymin']
    xmax = bounding_box['xmax']
    ymax = bounding_box['ymax']

    h = ymax - ymin
    w = xmax - xmin

    if h > w:
        border = (h - w) // 2
        # make sure that we don't go out of bounds TODO: we're still at risk of going out of bounds, but it is less likely
        if xmin - border < 0:
            xmax += border * 2
        elif xmax + border > 1:
            xmin -= border * 2
        else:
            xmin -= border
            xmax += border

    elif w > h:
        border = (w - h) // 2
        # make sure that we don't go out of bounds TODO: we're still at risk of going out of bounds, but it is less likely
        if ymin - border < 0:
            ymax += border * 2
        elif ymax + border > 1:
            ymin -= border * 2
        else:
            ymin -= border
            ymax += border

    bounding_box['xmin'] = xmin
    bounding_box['ymin'] = ymin
    bounding_box['xmax'] = xmax
    bounding_box['ymax'] = ymax

def preprocess_frame_for_detection(frame):
    """Preprocesses the frame for Yolo-v5 inference"""
    # right now, we are loading the Yolo-v5 model with Autoshape, so there isn't much to preprocess

    # convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def localize_fruit(frame: np.ndarray, bounding_boxes: List[dict], min_bounding_box_size) -> tf.Tensor:
    """Localizes the fruit in the frame and collects it into a batch of images for the disease and ripeness models
    Expects the bounding boxes to have normalized coordinates (0 to 1).
    Adds a 'small' field to the bounding boxes if the box is too small to localize."""
    h, w, _ = frame.shape

    localized_fruits = []

    for bounding_box in bounding_boxes:

        # make sure that the bounding box is large enough
        if bounding_box['xmax'] - bounding_box['xmin'] < min_bounding_box_size or bounding_box['ymax'] - bounding_box['ymin'] < min_bounding_box_size:
            bounding_box['small'] = True # if it isn't large enough, mark it as ignored
            continue
        else:
            bounding_box['small'] = False

        # get the bounding box coordinates
        xmin = int(bounding_box['xmin'] * w)
        ymin = int(bounding_box['ymin'] * h)
        xmax = int(bounding_box['xmax'] * w)
        ymax = int(bounding_box['ymax'] * h)

        # get the cropped image
        cropped_image = frame[ymin:ymax, xmin:xmax]

        # resize the image to the size that the disease and ripeness models expect
        cropped_image = cv2.resize(cropped_image, LOCALIZED_IMAGE_SIZE)

        # normalize the image and convert to float32
        cropped_image = cropped_image.astype(np.float32)

        # add to the list of localized fruits
        localized_fruits.append(cropped_image)

    # stack all of the images into a single tensor
    localized_fruits = tf.stack(localized_fruits)

    return localized_fruits

def prepare_output_frame(input_frame, bounding_boxes, ui='confidence'):
    # PySimpleGUI expects BGR, but DeepFruitVision uses RGB, so we have to convert the input frame to BGR
    frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape

    # loop through the bounding boxes and draw them
    for box in bounding_boxes:

        # get the bounding box coordinates
        xmin = box['xmin'] * w
        ymin = box['ymin'] * h
        xmax = box['xmax'] * w
        ymax = box['ymax'] * h

        ripeness_pred = box['ripeness']
        defect_pred = box['defect']

        if ui == 'confidence':
            if box['small']:
                color = (0, 0, 0)
            else:
                color = (0, 0, 255)

            # draw the bounding box
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            # display the class name and corresponding confidence
            cv2.putText(frame, f'{box["class"]} {box["conf"]:.2f}', (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # don't display ripeness or disease if the bounding box is too small
            if box['small']:
                continue

            # get the ripeness and disease predictions
            ripeness_class = ripeness.classnames[ripeness_pred[0]]
            ripeness_confidence = ripeness_pred[1]

            disease_class = defect.classnames[defect_pred[0]]
            disease_confidence = defect_pred[1]

            cv2.putText(frame, f'{ripeness_class} {ripeness_confidence:.2f}', (int(xmin), int(ymin) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f'{disease_class} {disease_confidence:.2f}', (int(xmin), int(ymin) - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif ui == 'harvestability':
            # don't display anything if the bounding box is too small
            if box['small']:
                continue

            if box['harvestability'] == 0: # if the fruit is not harvestable, draw a red box
                color = (0, 0, 255)
            elif box['harvestability'] == 1: # if the fruit is not yet ripe, draw a yellow box
                color = (0, 255, 255)
            else: # if the fruit is ripe, draw a green box
                color = (0, 255, 0)

            # draw the bounding box
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            # draw the name of the fruit
            cv2.putText(frame, f'{box["class"]}', (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


