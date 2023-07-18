class ExtractedLandmarks:
    hands_landmarks = []
    face_landmarks = []
    pose_landmarks = []

    def __init__(self, hands_landmarks, face_landmarks, pose_landmarks):
        self.hands_landmarks = hands_landmarks
        self.face_landmarks = face_landmarks
        self.pose_landmarks = pose_landmarks
