from time import sleep

import cv2
import pandas as pd
import numpy as np

# df = pd.read_csv("dataset_output/libras_ufop/libras_ufop_openpose.csv")
# df = pd.read_csv("dataset_output/libras_minds/libras_minds_openpose_80_frames.csv")
df = pd.read_csv("dataset_output/libras_minds/libras_minds_openpose.csv")
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

video_size = (1080, 1920, 3)
done_categories = []


def mirror_flip_landmarks_horizontally(df):
    # Mirror flip the x-coordinates horizontally
    return -df.values[::-1]

show_video = True
show_landmarks = False
frame_to_save = 50
frame_name = "frame_image_no_landmarks.png"

for video in videos:
    video_frames = df[df["video_name"] == video]
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("hand_0")]))
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("face_")]))
    # print(len([i for i in list(video_frames[video_frames["frame"] == 1].columns) if i.startswith("pose_")]))

    category = video_frames.iloc[0]["category"]
    video_name = video_frames.iloc[0]["video_name"]
    cap = cv2.VideoCapture(f"D:\\Projects\\datasets\\libras_minds\\raw\\{video_name}")
    x_columns = [i for i in video_frames.columns if i.endswith("_x")]
    # video_frames[x_columns] = mirror_flip_landmarks_horizontally(video_frames[x_columns])
    if category in done_categories:
        continue
    done_categories.append(category)
    # image = np.zeros(video_size, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = -1

    for index, frame in video_frames.iterrows():
        frame_count += 1
        image = np.ones(video_size, np.float32)
        # cv2.rectangle(image, (0, 0), (video_size[0], video_size[1]), color=(255, 255, 255), thickness=2000)
        if show_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, image = cap.read()
        org = (50, 50)

        # fontScale
        fontScale = 1

        if frame_count != frame_to_save:
            image = cv2.putText(image, str(frame_count), (100, 100), font,
                            3, (0, 0, 255), 1, cv2.LINE_AA)

        # Blue color in BGR

        # Line thickness of 2 px
        if show_landmarks:
            for lx, ly, lz in landmarks_name:
                character = "*"
                thickness = 1
                # color = (230, 53, 197, )
                color = (0, 255, 0,)
                if lx.startswith("face_"):
                    character = "."
                    color = (0, 255, 255)
                    thickness = 2
                if lx.startswith("pose_"):
                    character = "x"
                    color = (0, 0, 255)
                    thickness = 2
                if frame[lx] < 1 and frame[ly] < 1:
                    x = int(frame[lx] * video_size[1])
                    y = int(frame[ly] * video_size[0])
                    image = cv2.putText(image, character, (x, y), font,
                                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(f"image_{video}", image)
        if frame_count == frame_to_save:
            cv2.imwrite(frame_name, image)
        cv2.waitKey(33)
    cv2.destroyAllWindows()
    sleep(2)
