import cv2
import numpy as np
import tensorflow as tf
from typing import List

MIN_BOUNDING_BOX_SIZE = 0.15
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

def localize_fruit(frame: np.ndarray, bounding_boxes: List[dict]) -> tf.Tensor:
    """Localizes the fruit in the frame and collects it into a batch of images for the disease and ripeness models
    Expects the bounding boxes to have normalized coordinates (0 to 1) instead of pixel coordinates"""
    h, w, _ = frame.shape

    localized_fruits = []

    for bounding_box in bounding_boxes:
        # make sure that the bounding box is square
        square_bounding_box(bounding_box)

        # make sure that the bounding box is large enough
        if bounding_box['xmax'] - bounding_box['xmin'] < MIN_BOUNDING_BOX_SIZE or bounding_box['ymax'] - bounding_box['ymin'] < MIN_BOUNDING_BOX_SIZE:
            pass #TODO: right now, keep all of the bounding boxes that are too small, but later on, we'll want to remove them

        # get the bounding box coordinates
        xmin = bounding_box['xmin'] * w
        ymin = bounding_box['ymin'] * h
        xmax = bounding_box['xmax'] * w
        ymax = bounding_box['ymax'] * h

        # get the cropped image
        cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

        # resize the image to the size that the disease and ripeness models expect
        cropped_image = cv2.resize(cropped_image, LOCALIZED_IMAGE_SIZE)

        # add to the list of localized fruits
        localized_fruits.append(cropped_image)

    # stack all of the images into a single tensor
    localized_fruits = tf.stack(localized_fruits)

    return localized_fruits


