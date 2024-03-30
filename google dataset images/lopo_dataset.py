import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PIL import Image
import math


class LopoDataset(Dataset):
    def __init__(self, dataframe, frames, transforms=None, augment=True, person_in=[], person_out=[]):
        # self.df = dataframe
        # self.data = dataframe.values.astype(float)
        # self.scaler = StandardScaler()
        self.transforms = transforms

        # self.data = self.scaler.fit_transform(self.data)
        self.frames = frames
        self.augment = augment
        self.person_in = person_in
        self.person_out = person_out
        self.signs = self.get_signs(dataframe)
        self.categories = list(dataframe["category"].unique())
        self.df = self.prepare_data(dataframe)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        df = self.df.iloc[idx]
        category = df["category"]
        x = df["x"]
        y = df["y"]
        if self.augment:
            x, y = self.perform_augmentation(x, y)
        image = self.landmarks_to_image(x, y)
        image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
        # image = torch.tensor(image, dtype=torch.float32)
        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(self.categories.index(category), dtype=torch.int64)
        # return torch.tensor(image, dtype=torch.float32)

    def debug(self, idx, augment=True):
        df = self.df.iloc[idx]
        category = df["category"]
        x = df["x"]
        y = df["y"]
        if augment:
            x, y = self.perform_augmentation(x, y)
        return x, y, category

    def get_signs(self, df):
        signs = list(df.columns)
        signs = [s for s in signs if s.endswith("_x") or s.endswith("_y")]
        excluded_body_landmarks = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
        excluded_body_landmarks = tuple([f"pose_{i}" for i in excluded_body_landmarks])
        unwanted_pose_columns = [i for i in list(signs) if i.startswith(excluded_body_landmarks)]
        signs = [s for s in signs if s not in unwanted_pose_columns]
        return signs

    def get_axis_df(self, df, axis):
        return df[[c for c in self.signs if c.endswith(axis)]]

    def reshape_features_dataset(self, features):
        return features.reshape((int(features.shape[0]/self.frames), self.frames, features.shape[1]))

    def reshape_target_dataset(self, target):
        return target.reshape((int(target.shape[0]/self.frames), self.frames))[:, 0]

    def landmarks_to_image(self, x, y, n=3, normalize=False):
        # Remove cols until the width is multiple of n
        width = x.shape[1]
        if width % n != 0:
            extra_cols = width % n
            x = x[:, : width - extra_cols]
            y = y[:, : width - extra_cols]

        X = np.reshape(x, (x.shape[0], -1, n))
        Y = np.reshape(y, (y.shape[0], -1, n))
        I = np.concatenate([X, Y], axis=1)
        return I

    def normalize_axis(self, axis):
        axis[axis < 0] = 0
        axis[axis > 1] = 1
        return axis

    def prepare_data(self, df):
        columns = ["category", "video_name", "person", "frame"] + self.signs
        df = df[columns]
        videos = df["video_name"].unique()
        data = []
        for video in videos:
            df_video = df[df["video_name"] == video].sort_values("frame")
            category = df_video.iloc[0]["category"]
            person = df_video.iloc[0]["person"]
            if len(video) > self.frames:
                print("Video > frames")
                break
            if len(self.person_in) > 0:
                if person not in self.person_in:
                    continue
            if len(self.person_out) > 0:
                if person in self.person_out:
                    continue
            df_video = df_video.drop(["category", "video_name", "frame"], axis=1)
            x = self.get_axis_df(df_video, "x")
            y = self.get_axis_df(df_video, "y")
            x = x.T.to_numpy()
            y = y.T.to_numpy()
            x = self.normalize_axis(x)
            y = self.normalize_axis(y)
            # image = self.landmarks_to_image(x, y)
            data.append({
                "video_name": video,
                "x": x,
                "y": y,
                "category": category
            })
        return pd.DataFrame.from_dict(data)

    def rotate_landmarks(self, x, y, rotation_angle):
        # Assuming landmarks are in (x, y, z) format
        landmarks = np.column_stack((x.ravel(), y.ravel()))

        # Calculate the centroid (center) of the landmarks
        #     centroid = np.mean(landmarks, axis=0)
        centroid = 0.5

        # Translate landmarks to the origin (center)
        translated_landmarks = landmarks - centroid
        # Create a rotation matrix for the given angle (in radians)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

        # Apply the rotation to each landmark
        #     print(rotation_matrix)
        rotated_landmarks = np.dot(translated_landmarks, rotation_matrix) + centroid

        x_rotated = rotated_landmarks[:, 0].reshape(x.shape)
        y_rotated = rotated_landmarks[:, 1].reshape(y.shape)

        return x_rotated, y_rotated

    def zoom_landmarks(self, landmarks, zoom_factor):
        # Scale the landmarks by the zoom factor
        zoomed_landmarks = landmarks * zoom_factor

        return zoomed_landmarks

    def translate_landmarks(self, landmarks, translation_vector):
        # Translate the landmarks by the given vector
        translated_landmarks = landmarks + translation_vector

        return translated_landmarks

    def apply_transformation(self, x, y, rotation, zoom, translate_x, translate_y):
        x, y = self.rotate_landmarks(x, y, math.radians(rotation))
        x = self.zoom_landmarks(x, zoom)
        x = self.translate_landmarks(x, [translate_x])

        # y = self.rotate_landmarks_y(y, math.radians(rotation))
        y = self.zoom_landmarks(y, zoom)
        y = self.translate_landmarks(y, [translate_y])
        return x, y

    def perform_augmentation(self, x, y):
        rotation_sigma = 4
        zoom_sigma = 0.05
        translate_x_sigma = 0.01
        translate_y_sigma = 0.0
        rotation = np.random.normal(0, rotation_sigma)
        zoom = np.random.normal(0, zoom_sigma) + 1
        translate_x = np.random.normal(0, translate_x_sigma)
        translate_y = np.random.normal(0, translate_y_sigma)
        x, y = self.apply_transformation(x, y, rotation, zoom, translate_x, translate_y)
        return x, y

