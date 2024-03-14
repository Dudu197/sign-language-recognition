import cv2
import pandas as pd
import numpy as np

# df = pd.read_csv("dataset_output/libras_minds/libras_minds_hello.csv")
# df = pd.read_csv("datasets/libras_minds_dataset_openpose.csv")
df = pd.read_csv("datasets/dot/daniele/Abias.MOV.csv")
# print(df)

excluded_body_landmarks = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
excluded_body_landmarks = tuple([f"pose_{i}" for i in excluded_body_landmarks])

landmarks_name = [i for i in list(df.columns) if i not in ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "category", "category_index", "video_name", "frame", "missing_hand", "missing_face"]]
landmarks_name = np.array([i for i in landmarks_name if not i.startswith(excluded_body_landmarks)])
landmarks_name = landmarks_name.reshape((int(landmarks_name.shape[0]/3), 3))
videos = df["video_name"].unique()

video_size = (1280, 720)
done_categories = []


for video in videos:
    video_frames = df[df["video_name"] == video]
    category = video_frames.iloc[0]["category"]
    if category in done_categories:
        continue
    done_categories.append(category)
    for index, frame in video_frames.iterrows():
        img = np.zeros(video_size, np.float32)
        for lx, ly, lz in landmarks_name:
            if frame[lx] < 1 and frame[ly] < 1:
                x = int(frame[lx] * video_size[1])
                y = int(frame[ly] * video_size[0])
                img[y, x] = 1
        cv2.imshow(f"image_{video}", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(33)
    cv2.destroyAllWindows()
