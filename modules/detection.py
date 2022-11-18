import numpy as np
import torch

from typing import List

classnames = ['apple', 'papaya', 'mango']

class FruitDetectionModule(object):
    def __init__(self, weights_path, device = 'cuda:0'):
        self.model = torch.hub.load('yolov5', 'custom', path=weights_path,
                                    source='local', device=device)

    def get_bounding_boxes(self, img: np.array) -> List[dict]:
        """Generates bounding boxes around every fruit in the image. Returns a list of bounding boxes.
        Each row is a bounding box, represented by a dictionary with keys {'xmin', 'ymin', 'xmax', 'ymax', 'name'}
        The x and y values are  normalized values like the input to Yolo-v5. Name refers to the class name (e.g. 'apple')
        """

        predictions = self.model(img)
        bounding_boxes = []

        # the first dimension of predictions.xyxy is the batch dimension, and since we only have one image,
        # we can ignore it and loop through all bounding boxes in the image
        for prediction in predictions.xyxyn[0]:
            prediction = prediction.detach().cpu().numpy()

            xmin = prediction[0]
            ymin = prediction[1]
            xmax = prediction[2]
            ymax = prediction[3]

            conf = prediction[4]

            class_index = int(prediction[5])
            class_name = self.model.names[class_index]

            bounding_boxes.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'conf': conf,
                'class': class_name
            })

        return bounding_boxes