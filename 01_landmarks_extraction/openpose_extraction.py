import cv2
import numpy as np
import time
from openpose import OpenPoseExtractor
import pandas as pd
import os
# import sys
import re


# category_to_process = int(sys.argv[1])
# category_to_process = sys.argv[1]
# category_to_process = "Acontecer"
max_num_hands = 2

# base_path = "Videos/Videos"
base_path = "D:\\Projects\\datasets\\daniele"
# categories = os.listdir(base_path)
# categories = [str(i + 1).ljust(2, "0") for i in range(20)]

videos_data = []

extractor = OpenPoseExtractor()

def get_video_landmarks(video_path):
    video = cv2.VideoCapture(video_path)
    extracted_landmarks = []
    while video.isOpened():
        check, img = video.read()
        if check:
            landmarks = extractor.extract_image(img)
            extracted_landmarks.append(landmarks)
        else:
            break
    return extracted_landmarks


def extract_hands(hand):
    landmarks = {}
    hand_count = 0
    if hand is not None:
        for h in hand:
            if hand_count >= max_num_hands:
                break
            landmark_count = 0
            for l in h[0]:
                landmarks[f"hand_{hand_count}_{landmark_count}_x"] = l[0]
                landmarks[f"hand_{hand_count}_{landmark_count}_y"] = l[1]
                landmarks[f"hand_{hand_count}_{landmark_count}_z"] = l[2]
                landmark_count += 1
            hand_count += 1
    for h in range(hand_count, max_num_hands):
        for l in range(21):
            landmarks[f"hand_{h}_{l}_x"] = 0
            landmarks[f"hand_{h}_{l}_y"] = 0
            landmarks[f"hand_{h}_{l}_z"] = 0
    return landmarks


def extract_face(face):
    landmarks = {}
    if face is not None:
        landmark_count = 0
        if type(face) in [list, np.ndarray]:
            if len(face) > 2:
                # raise "Não era para ser"
                print("More than one face detected. Using first face")
            face = face[0]
        for l in face:
            landmarks[f"face_{landmark_count}_x"] = l[0]
            landmarks[f"face_{landmark_count}_y"] = l[1]
            landmarks[f"face_{landmark_count}_z"] = l[2]
            landmark_count += 1
    else:
        for l in range(70):
            landmarks[f"face_{l}_x"] = 0
            landmarks[f"face_{l}_y"] = 0
            landmarks[f"face_{l}_z"] = 0
    return landmarks


def extract_pose(pose):
    landmarks = {}
    if pose is not None:
        landmark_count = 0
        for l in pose[0]:
            landmarks[f"pose_{landmark_count}_x"] = l[0]
            landmarks[f"pose_{landmark_count}_y"] = l[1]
            landmarks[f"pose_{landmark_count}_z"] = l[2]
            landmark_count += 1
    else:
        for l in range(25):
            landmarks[f"pose_{l}_x"] = 0
            landmarks[f"pose_{l}_y"] = 0
            landmarks[f"pose_{l}_z"] = 0
    return landmarks


def process_video(video_path, category, category_index, video_name, signaler):
    print(f"Processing {category} - {video_name}")
    landmarks = get_video_landmarks(video_path)
    frame_count = 0
    for l in landmarks:
        frame_data = {
            "category": category_index,
            "video_name": video_name,
            "frame": frame_count,
            "person": signaler
        }
        frame_data.update(extract_hands(l.hands_landmarks))
        frame_data.update(extract_face(l.face_landmarks))
        frame_data.update(extract_pose(l.pose_landmarks))
        frame_count += 1
        videos_data.append(frame_data)
    return frame_count

# videos_to_process = []
# for category_index in range(len(categories)):
#     category = categories[category_index]
#     videos = os.listdir(os.path.join(base_path, category))
#     for video in videos:
#         process_video(base_path, category, video)


# for base_category in os.listdir(base_path):
#     if base_category != category_to_process or "." in base_category:
#         continue
#     for category in os.listdir(os.path.join(base_path, base_category)):
#         for video in os.listdir(os.path.join(base_path, base_category, category)):
#             video_path = os.path.join(base_category, category, video)
#             signaler = 0
#             videos_to_process.append((video_path, category, f"{base_category}_{signaler}", signaler))



# for video in os.listdir(base_path):
# for video in os.listdir(os.path.join(base_path, category)):
#     category_name, index = video.replace(".avi", "").split("_")
#     category = int(video[:2])
#     signaler = (int((int(index) + 1) / 15))
#     result = re.findall(r"(\d\d)(.*)Sinalizador(\d\d)", video)[0]
#     category = result[1]
#     signaler = int(result[2])
    # category = int(category)
    # if category > category_to_process:
    #     break
    # if category < category_to_process - 1:
    # if category_to_process not in video:
    #     continue
    # videos_to_process.append((video, 0, 0, 0))

def process(videos_to_process):
    processed = 0
    videos_len = len(videos_to_process)
    total_start_time = time.time()
    for video, category, signaler, index in videos_to_process:
        processed += 1
        start_time = time.time()
        frame_count = process_video(os.path.join(base_path, video), category, category, video, signaler)
        end_time = time.time()
        duration = end_time - start_time
        eta = duration * (videos_len - processed)
        print(f"Benchmark: duration: {duration}; frames: {frame_count}")
        print(f"Duration: {duration}s. ETA: {int(eta/60)}m")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total execution time: {int(total_time/60)}m or {total_time}s")
    df = pd.DataFrame(videos_data)
    return df

videos_to_process = [(os.path.join(base_path, i), i, 0, 0) for i in os.listdir(base_path)]
df = process(videos_to_process)
df.to_csv("../00_datasets/dataset_output/daniele/daniele_openpose.csv", index=False)
