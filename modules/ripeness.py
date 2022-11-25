import numpy as np
import tensorflow as tf

from typing import List, Tuple

classnames = ['unripe', 'ripe']

class RipenessModule(object):
    def __init__(self, weights_path, device='/GPU:0'):
        self.device = device

        self.model = tf.keras.models.load_model(weights_path)

    def get_ripeness_predictions(self, img_batch: tf.Tensor) -> List[Tuple[int, float]]:
        # if there are no images in the input batch, we just return an empty list
        if img_batch.shape[0] > 0:
            predictions = self.model(img_batch, training=False)

            ripeness_predictions = [(np.argmax(prediction), np.max(prediction)) for prediction in predictions]

            return ripeness_predictions
        else:
            return []