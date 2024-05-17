import cv2
import mediapipe as mp
import pandas as pd
import json
import os
from extracted_landmarks import ExtractedLandmarks


hand_positions = [
    'wrist',

    'thumb_cmc',
    'thumb_mcp',
    'thumb_ip',
    'thumb_tip',

    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',

    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',

    'ring_pip',
    'ring_pip',
    'ring_dip',
    'ring_tip',

    'pinky_mcp',
    'pinky_pip',
    'pinky_dip',
    'pinky_tip',
]

pose_positions = [
    "nose",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_ear",
    "left_ear",
    "mouth_right",
    "mouth_left",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
    "right_pinky_1",
    "left_pinky_1",
    "right_index_1",
    "left_index_1",
    "right_thumb_1",
    "left_thumb_1",
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
    "right_heel",
    "left_heel",
    "right_foot_index",
    "left_foot_index",
]

max_num_hands = 2
min_detection_confidence = 0.4

# base_path = "Videos/Videos"
base_path = "D:\\Projects\\datasets\\daniele"
# categories = os.listdir(base_path)
categories = [i for i in range(64)]
videos_data = []


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
hand = mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)
face = mp_face.FaceMesh(min_detection_confidence=min_detection_confidence)
pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence)

def get_video_landmarks(video_path):
    video = cv2.VideoCapture(video_path)
    extracted_landmarks = []
    while video.isOpened():
        check, img = video.read()
        if check:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hand_result = hand.process(imgRGB)
            face_result = face.process(imgRGB)
            pose_result = pose.process(imgRGB)
            landmarks = ExtractedLandmarks(hand_result.multi_hand_landmarks, face_result.multi_face_landmarks, pose_result.pose_landmarks)
            extracted_landmarks.append(landmarks)
        else:
            break
    return extracted_landmarks


def extract_hands(hand):
    landmarks = {}
    hand_count = 0
    if hand:
        for h in hand:
            if hand_count >= max_num_hands:
                break
            landmark_count = 0
            for l in h.landmark:
                landmarks[f"hand_{hand_count}_{hand_positions[landmark_count]}_x"] = l.x
                landmarks[f"hand_{hand_count}_{hand_positions[landmark_count]}_y"] = l.y
                landmarks[f"hand_{hand_count}_{hand_positions[landmark_count]}_z"] = l.z
                landmark_count += 1
            hand_count += 1
    for h in range(hand_count, max_num_hands):
        for l in range(len(hand_positions)):
            landmarks[f"hand_{h}_{hand_positions[l]}_x"] = 0
            landmarks[f"hand_{h}_{hand_positions[l]}_y"] = 0
            landmarks[f"hand_{h}_{hand_positions[l]}_z"] = 0
    return landmarks


def extract_face(face):
    landmarks = {}
    if face:
        landmark_count = 0
        if type(face) == list:
            if len(face) > 1:
                raise "Não era para ser"
            face = face[0]
        for l in face.landmark:
            landmarks[f"face_{landmark_count}_x"] = l.x
            landmarks[f"face_{landmark_count}_y"] = l.y
            landmarks[f"face_{landmark_count}_z"] = l.z
            landmark_count += 1
    else:
        for l in range(468):
            landmarks[f"face_{l}_x"] = 0
            landmarks[f"face_{l}_y"] = 0
            landmarks[f"face_{l}_z"] = 0
    return landmarks


def extract_pose(pose):
    landmarks = {}
    if pose:
        landmark_count = 0
        for l in pose.landmark:
            landmarks[f"pose_{pose_positions[landmark_count]}_x"] = l.x
            landmarks[f"pose_{pose_positions[landmark_count]}_y"] = l.y
            landmarks[f"pose_{pose_positions[landmark_count]}_z"] = l.z
            landmark_count += 1
    else:
        for l in range(len(pose_positions)):
            landmarks[f"pose_{pose_positions[l]}_x"] = 0
            landmarks[f"pose_{pose_positions[l]}_y"] = 0
            landmarks[f"pose_{pose_positions[l]}_z"] = 0
    return landmarks


def process_video(video_path, category, category_index, video_name):
    print(f"Processing {category} - {video_name}")
    landmarks = get_video_landmarks(video_path)
    frame_count = 0
    for l in landmarks:
        frame_data = {
            "category": category_index,
            "video_name": video_name,
            "frame": frame_count
        }
        frame_data.update(extract_hands(l.hands_landmarks))
        frame_data.update(extract_face(l.face_landmarks))
        frame_data.update(extract_pose(l.pose_landmarks))
        frame_count += 1
        videos_data.append(frame_data)


# for category_index in range(len(categories)):
#     category = categories[category_index]
#     videos = os.listdir(os.path.join(base_path, category))
#     for video in videos:
#         process_video(base_path, category, video)

for video in os.listdir(base_path):
    category = video.replace(".mp4", "")
    signaler = 1
    # if category > 30:
    #     break
    # if category <= 20:
    #     continue
    process_video(os.path.join(base_path, video), category, category, video)

    df = pd.DataFrame(videos_data)
    df.to_csv(f"datasets/dot/daniele/{category}.csv")
    videos_data = []
