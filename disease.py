import numpy as np
import tensorflow as tf

from typing import List


class DiseaseModule(object):
    def __init__(self, weights_path, device='/GPU:0'):
        self.device = device
        self.classnames = ['unhealthy', 'healthy']

        with tf.device(device):
            self.model = tf.keras.models.load_model(weights_path)

    def get_disease_predictions(self, img_batch: tf.Tensor) -> List[str]:
        # if there are no images in the input batch, we just return an empty list
        if img_batch.shape[0] > 0:
            with tf.device(self.device):
                predictions = self.model(img_batch)

            disease_predictions = []
            for prediction in predictions:
                disease_predictions.append(self.classnames[np.argmax(prediction)])

            return disease_predictions
        else:
            return []