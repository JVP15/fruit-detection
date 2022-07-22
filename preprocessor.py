import cv2

MIN_BOUNDING_BOX_SIZE = 0.15

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

