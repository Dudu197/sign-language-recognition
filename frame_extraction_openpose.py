import cv2 as cv
import pandas as pd
from openpose import OpenPoseExtractor
import cv2


base_path = "Videos/Videos"

def get_video_landmarks(video_path):
    video = cv2.VideoCapture(video_path)
    extracted_landmarks = []
    while video.isOpened():
        check, img = video.read()
        if check:
            result = OpenPoseExtractor.extract_image(img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hand_result = hand.process(imgRGB)
            face_result = face.process(imgRGB)
            pose_result = pose.process(imgRGB)
            landmarks = ExtractedLandmarks(hand_result.multi_hand_landmarks, face_result.multi_face_landmarks, pose_result.pose_landmarks)
            extracted_landmarks.append(landmarks)
        else:
            break
    return extracted_landmarks

