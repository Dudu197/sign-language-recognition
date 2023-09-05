import cv2
import pandas as pd
import numpy as np

df = pd.read_csv("lsa64_dataset_best_80_fps.csv")
# print(df)

landmarks_name = np.array([i for i in list(df.columns) if i not in ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "category", "category_index", "video_name", "frame", "missing_hand", "missing_face"]])
landmarks_name = landmarks_name.reshape((int(landmarks_name.shape[0]/3), 3))
videos = df["video_name"].unique()

video_size = (720, 1080)
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
