from .base_image_representation import BaseImageRepresentation
import numpy as np


class SkeletonDMLRepresentation(BaseImageRepresentation):
    name = "Skeleton-DML"

    def transform(self, x, y, z):
        n = 3
        width = x.shape[1]
        if width % n != 0:
            extra_cols = width % n
            x = x[:, : width - extra_cols]
            y = y[:, : width - extra_cols]

        x = np.reshape(x, (x.shape[0], -1, n))
        y = np.reshape(y, (y.shape[0], -1, n))
        image = np.concatenate([x, y], axis=1)
        return image
