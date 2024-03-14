from openpose import OpenPoseExtractor
import cv2


video = cv2.VideoCapture(r"D:\Projects\datasets\LSA64\001_001_002.mp4")
while video.isOpened():
    check, img = video.read()
    if check:
        result = OpenPoseExtractor().extract_image(img)
        print(result)
    else:
        break
