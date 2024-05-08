from time import sleep

import cv2
import pandas as pd
import numpy as np

# df = pd.read_csv("dataset_output/libras_ufop/libras_ufop_openpose.csv")
# df = pd.read_csv("dataset_output/libras_minds/libras_minds_openpose_80_frames.csv")
df = pd.read_csv("dataset_output/include50/include50_openpose.csv")
# df = pd.read_csv("datasets/libras_minds_dataset_openpose.csv")
# df = pd.read_csv("datasets/dot/daniele/Abias.MOV.csv")
# print(df)

excluded_body_landmarks = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
excluded_body_landmarks = tuple([f"pose_{i}" for i in excluded_body_landmarks])

landmarks_name = [i for i in list(df.columns) if i not in ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "category", "category_index", "video_name", "frame", "missing_hand", "missing_face", "person"]]
# landmarks_name = [i for i in landmarks_name if i.startswith("hand_")]
landmarks_name = np.array([i for i in landmarks_name if not i.startswith(excluded_body_landmarks)])
landmarks_name = landmarks_name.reshape((int(landmarks_name.shape[0]/3), 3))
videos = df["video_name"].unique()

video_size = (720, 1280)
done_categories = []


def mirror_flip_landmarks_horizontally(df):
    # Mirror flip the x-coordinates horizontally
    return -df.values[::-1]


for video in videos:
    video_frames = df[df["video_name"] == video]
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("hand_0")]))
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("face_")]))
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("pose_")]))

    category = video_frames.iloc[0]["category"]
    x_columns = [i for i in video_frames.columns if i.endswith("_x")]
    # video_frames[x_columns] = mirror_flip_landmarks_horizontally(video_frames[x_columns])
    if category in done_categories:
        continue
    done_categories.append(category)
    for index, frame in video_frames.iterrows():
        img = np.ones(video_size, np.float32)
        for lx, ly, lz in landmarks_name:
            if frame[lx] < 1 and frame[ly] < 1:
                x = int(frame[lx] * video_size[1])
                y = int(frame[ly] * video_size[0])
                if y == 719:
                    y = 718
                if x == 1279:
                    x = 1278
                img[y, x] = 0
                img[y-1, x] = 0
                img[y+1, x] = 0
                img[y, x-1] = 0
                img[y, x+1] = 0
                img[y-1, x+1] = 0
                img[y-1, x-1] = 0
                img[y+1, x+1] = 0
                img[y+1, x-1] = 0
        cv2.imshow(f"image_{video}", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(33)
    cv2.destroyAllWindows()
    # sleep(30)
