from skeleton_magnitude_representation import SkeletonMagnitudeRepresentation
import numpy as np
from matplotlib import pyplot as plt

with open("data.npy", "rb") as f:
    x = np.load(f)
    y = np.load(f)
    z = np.load(f)

print(x.shape)
# 127 landmarks
# 104 frames
# s =  C × T × 3,
# C = Landmarks
# T = Frames
sk = SkeletonMagnitudeRepresentation([5, 10, 15])
image = np.array(sk.transform(x, y, z))
print(image.shape)
# image = np.moveaxis(image, [0, 1], [2, 1])
print(image.shape)
plt.imshow(image)
plt.show()
