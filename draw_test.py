import cv2
import pandas as pd
import numpy as np

# df = pd.read_csv("dataset_output/libras_minds/libras_minds_hello.csv")
df = np.load("../skeleton-dml/representation/a1_s1_t1_skeleton.npy")
# print(df)


video_size = (720, 1280)

for frame in df:
    img = np.zeros(video_size, np.float32)
    for l in frame:
        lx = l[0]
        ly = l[1]
        lz = l[2]
        if lx < 1 and ly < 1:
            x = int(lz * video_size[1])
            y = int(ly * video_size[0])
            img[y, x] = 1
    cv2.imshow(f"image", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(100)
cv2.destroyAllWindows()
