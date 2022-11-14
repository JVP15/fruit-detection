from modules.detection import FruitDetectionModule
from modules.ripeness import RipenessModule
from modules.defect import DefectModule

import modules.preprocessor as preprocessor

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

        def add_ripeness_and_defect(bounding_box):
            nonlocal prediction_index
            if bounding_box['small']:
                bounding_box['ripeness'] = None
                bounding_box['defect'] = None
            else:
                bounding_box['ripeness'] = ripeness_predictions[prediction_index]
                bounding_box['defect'] = defect_predictions[prediction_index]
                prediction_index += 1

            return bounding_box

        # add the ripeness and defect predictions to the bounding boxes
        bounding_boxes = list(map(add_ripeness_and_defect, bounding_boxes))

        return bounding_boxes
