from .base_image_representation import BaseImageRepresentation
import numpy as np


class SlDMLRepresentation(BaseImageRepresentation):
    name = "SL-DML"

    def transform(self, x, y, z):
        landmarks = np.stack((x, y, z), axis=1)
        t = np.swapaxes(landmarks, 1, 2)
        t = np.concatenate([x, y, x], axis=1)
        # Normalization
        t -= np.min(t)
        t /= np.max(t)
        return t
