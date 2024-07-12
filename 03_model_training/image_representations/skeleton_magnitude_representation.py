from base_image_representation import BaseImageRepresentation
import numpy as np


class SkeletonMagnitudeRepresentation(BaseImageRepresentation):
    name = "Skeleton Magnitude"

    def __init__(self, temporal_scales):
        self.temporal_scales = temporal_scales

    def transform(self, x, y, z):
        # img = np.zeros((x.shape[0], x.shape[1], len(self.temporal_scales)), np.float)

        mag_values = []
        for t_scale in self.temporal_scales:
            diff_joint = np.array(self.compute_temporal_joint_difference(x, y, z, t_scale))
            # diff_joint = np.moveaxis(diff_joint, [0], [2])
            mag_values.append(self.compute_joint_magnitude(diff_joint))
        img = np.array(mag_values)
        img = np.moveaxis(img, [0], [2])
        return img
        # img[j_pos, i_frames] = tuple(mag_values)

        return img

    def compute_temporal_joint_difference(self, x, y, z, temporal_dist: int) -> (float, float, float):
        shifted_x = np.roll(x, -temporal_dist)
        shifted_y = np.roll(y, -temporal_dist)
        shifted_z = np.roll(z, -temporal_dist)

        shifted_x[-temporal_dist:] = 0
        shifted_y[-temporal_dist:] = 0
        shifted_z[-temporal_dist:] = 0

        diff_x = x - shifted_x
        diff_y = y - shifted_x
        diff_z = z - shifted_x

        return diff_x, diff_y, diff_z

    # def compute_temporal_joint_difference(self, i_frame: int, j_joint: int, k_body: int, temporal_dist: int) -> (float, float, float):
    #     ret = (0.0, 0.0, 0.0)
    #     if (i_frame + temporal_dist) < self.kinect_data.n_frames and self.kinect_data.kinect_blocks[i_frame + temporal_dist].n_bodies > k_body:
    #         joint_data_f1 = self.kinect_data.kinect_blocks[i_frame].body_list[k_body].joint_data[j_joint]
    #         joint_data_f2 = self.kinect_data.kinect_blocks[i_frame + temporal_dist].body_list[k_body].joint_data[j_joint]
    #         diff_x = joint_data_f1.x_joint - joint_data_f2.x_joint
    #         diff_y = joint_data_f1.y_joint - joint_data_f2.y_joint
    #         diff_z = joint_data_f1.z_joint - joint_data_f2.z_joint
    #         ret = (diff_x, diff_y, diff_z)
    #     return ret

    def compute_joint_magnitude(self, diff_joint: np.array) -> float:
        ret = (diff_joint ** 2).sum(axis=0) ** (1. / 2)
        ret = self.normalize(ret, 0.0, 1, 1.0, 0.0)
        return ret

    def normalize(self, value: float, lower_bound: float, higher_bound: float, max_value: int, min_value: int) -> float:
        value[value > higher_bound] = max_value
        value[value < lower_bound] = min_value
        return value
        # if value > higher_bound:
        #     ret_value = max_value
        # elif value < lower_bound:
        #     ret_value = min_value
        # else:
        #     ret_value = (max_value * ((value - lower_bound) / (higher_bound - lower_bound)))  # estava com cast de int()
        # return ret_value
