import numpy as np
import tensorflow as tf

from typing import List, Tuple

classnames = ['defective', 'normal']

class DefectModule(object):
    def __init__(self, weights_path, device='/GPU:0'):
        self.device = device

        with tf.device(device):
            self.model = tf.keras.models.load_model(weights_path)

    def get_defect_predictions(self, img_batch: tf.Tensor) -> List[Tuple[int, float]]:
        # if there are no images in the input batch, we just return an empty list
        if img_batch.shape[0] > 0:
            with tf.device(self.device):
                predictions = self.model(img_batch)

            defect_predictions = [(np.argmax(prediction), np.max(prediction)) for prediction in predictions]

            return defect_predictions
        else:
            return []