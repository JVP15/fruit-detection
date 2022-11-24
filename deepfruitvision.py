from modules.detection import FruitDetectionModule
from modules.ripeness import RipenessModule
from modules.defect import DefectModule

import modules.preprocessor as preprocessor

classnames = ['defective fruit', 'unripe fruit', 'harvestable fruit']

def add_harvestability_to_box(bounding_box):
    """This function takes a bounding box that has already been processed by the detection, ripeness, and defect model
    and adds a 'harvestability' key to the bounding box dictionary. The value of the 'harvestability' key is an integer
    that represents the harvestability of the fruit.
    0 means the fruit is unharvestable because it is defective.
    1 means the fruit is not yet harvestable because it is unripe.
    2 means the fruit is harvestable.
    If the bounding box is too small, the 'harvestability' key is set to None because the fruit is too small to make an
    accurate prediction about its harvestability.
    """

    if bounding_box['small']:
        bounding_box['harvestability'] = None
    else:
        if bounding_box['defect'][0] == 0:
            bounding_box['harvestability'] = 0
        elif bounding_box['ripeness'][0] == 0:
            bounding_box['harvestability'] = 1
        else:
            bounding_box['harvestability'] = 2

    return bounding_box

class DeepFruitVision:
    def __init__(self, detection_weights, ripeness_weights, defect_weights, min_bounding_box_size=0.1):
        self.detection_module = FruitDetectionModule(detection_weights)
        self.ripeness_module = RipenessModule(ripeness_weights)
        self.defect_module = DefectModule(defect_weights)
        self.min_bounding_box_size = min_bounding_box_size

    def predict(self, frame):
        bounding_boxes = self.detection_module.get_bounding_boxes(frame)

        localized_fruit = preprocessor.localize_fruit(frame, bounding_boxes, self.min_bounding_box_size)

        ripeness_predictions = self.ripeness_module.get_ripeness_predictions(localized_fruit)
        defect_predictions = self.defect_module.get_defect_predictions(localized_fruit)

        # keep track of the index of the ripeness and disease predictions separately b/c some bounding boxes may be too small
        prediction_index = 0

        # add the ripeness and defect predictions to the bounding boxes
        for bounding_box in bounding_boxes:
            if bounding_box['small']:
                bounding_box['ripeness'] = None
                bounding_box['defect'] = None
            else:
                bounding_box['ripeness'] = ripeness_predictions[prediction_index]
                bounding_box['defect'] = defect_predictions[prediction_index]
                prediction_index += 1

        return bounding_boxes

    def get_harvestability(self, frame):
        """This function takes a frame and returns a list of bounding boxes (represented by a dictionary) that have a class,
        a bounding box, a ripeness prediction, defect prediction, and harvestability prediction.
        The bounding boxes are in the format 'xmin', 'ymin', 'xmax', 'ymax', with normalized values like yolo-v5, and they also have a corresponding 'conf' value.
        Each bounding box also has a 'small' key that is True if the bounding box is too small to make an accurate prediction about the harvestability
        of the fruit. In which case, the 'harvestability' key is set to None.
        """

        bounding_boxes = self.predict(frame)

        bounding_boxes = [add_harvestability_to_box(bounding_box) for bounding_box in bounding_boxes]

        return bounding_boxes


