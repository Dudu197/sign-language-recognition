import os
import sys
import cv2
from extracted_landmarks import ExtractedLandmarks


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = r"C:\Users\duduu\Downloads\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\python"
try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(dir_path + '/../bin/python/openpose/Release');
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenPoseExtractor:
    def __init__(self):
        params = dict()
        params[
            "model_folder"] = r"C:\Users\duduu\Downloads\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\models/"
        params["face"] = True
        params["hand"] = True
        params["net_resolution"] = "-1x256"
        params["face_net_resolution"] = "256x256"
        params["hand_net_resolution"] = "256x256"

        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

    def extract_image(self, image):
        # Process Image
        datum = op.Datum()
        datum.cvInputData = image
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

        landmarks = ExtractedLandmarks(
            hands_landmarks=OpenPoseExtractor.normalize_hand(datum.handKeypoints, image),
            face_landmarks=OpenPoseExtractor.normalize_face(datum.faceKeypoints, image),
            pose_landmarks=OpenPoseExtractor.normalize_pose(datum.poseKeypoints, image)
        )

        return landmarks

    @staticmethod
    def normalize_hand(hand, image):
        for h in hand:
            h[0][:, 0] = h[0][:, 0] / image.shape[1]
            h[0][:, 1] = h[0][:, 1] / image.shape[0]
        return hand

    @staticmethod
    def normalize_face(face, image):
        face[0][:, 0] = face[0][:, 0] / image.shape[1]
        face[0][:, 1] = face[0][:, 1] / image.shape[0]
        return face

    @staticmethod
    def normalize_pose(pose, image):
        pose[0][:, 0] = pose[0][:, 0] / image.shape[1]
        pose[0][:, 1] = pose[0][:, 1] / image.shape[0]
        return pose
