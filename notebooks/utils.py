import os
import random
import tempfile
import cv2
import numpy as np
import argparse
import shutil

# from skimage.util.shape import view_as_blocks
# from skimage.util import montage
from collections import defaultdict
from skimage.filters import threshold_sauvola


class defaultdict_factory(defaultdict):
    def __init__(self, factory_func):
        super().__init__(None)
        self.factory_func = factory_func

    def __missing__(self, key):
        ret = self.factory_func(key)
        self[key] = ret

        return self[key]


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def clear_dir(path):
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            os.remove("{}/{}".format(root, fname))


def remove_dir(path):
    shutil.rmtree(path, ignore_errors=True)


def move_dir(src, dst=None):
    if dst is None:
        dst = tempfile.NamedTemporaryFile().name
    try:
        _ = shutil.move(src, dst)
    except FileNotFoundError as exc:
        pass


def load_rgb(fname):

    rgb = cv2.imread(fname)[..., ::-1]
    return rgb


def save_rgb(image, fname):
    assert image.ndim == 3

    bgr = image[..., ::-1]
    cv2.imwrite(fname, bgr)  # it saves rgb, so we have to invert before save


def load_grayscale(fname, num_channels=1):
    assert num_channels in [1, 3]

    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if num_channels == 3:
        gray = np.stack(3 * [gray]).transpose(1, 2, 0)
    return gray


def load_binary(fname, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]

    gray = load_grayscale(fname)
    binary = grayscale_to_binary(gray, thresh_func)
    thresh = thresh_func(gray)
    binary = (255 * (binary > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


def crop_central_area(image, w_crop):
    _, w = image.shape[:2]
    x_start = (w - w_crop) // 2
    return image[:, x_start : x_start + w_crop].copy()


def rgb_to_grayscale(image):
    assert image.ndim == 3

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def grayscale_to_rgb(image):
    assert image.ndim == 2

    rgb = np.stack([image, image, image]).transpose((1, 2, 0))
    return rgb


def grayscale_to_binary(image, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]
    assert image.ndim == 2

    thresh = thresh_func(image)
    binary = (255 * (image > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


def rgb_to_binary(image, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]
    assert image.ndim == 3

    gray = rgb_to_grayscale(image)
    binary = grayscale_to_binary(gray, thresh_func, num_channels)
    return binary


def compute_threshold(image, thresh_func=threshold_sauvola):
    assert image.ndim in [2, 3]

    if image.ndim == 3:
        image = rgb_to_grayscale(image)
    return thresh_func(image)


def apply_threshold(image, thresh, num_channels=1):
    assert image.ndim in [2, 3]
    if image.ndim == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image
    binary = (255 * (gray > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


def is_integer(value):
    return value == int(value)


def doc_id_from_path(fname):
    doc_without_ext = os.path.splitext(fname)[0]
    doc_basename = os.path.basename(doc_without_ext)
    return doc_basename


def save_list_as_txt(list_, fname, shuffle=False):
    list_cpy = list_.copy()
    if shuffle:
        random.shuffle(list_cpy)
    txt = "\n".join(list_cpy)
    open(fname, "w").write(txt)


def decode_txt(fname, num_cols=2):
    assert num_cols > 1
    txt = []
    lines = open(fname, "r").readlines()
    for line in lines:
        line_split = line.strip().split()[:num_cols]
        txt.append(line_split)
    cols = [list(col) for col in zip(*txt)]
    return cols


def sample_from_lists(*lists, k=10):
    lists_merged = list(zip(*lists))
    lists_sampled = random.sample(lists_merged, k=k)
    return list(zip(*lists_sampled))


def merge_by_keys(dict_, f_merge):
    """Merge entries with similar keys, where similarity is defined by f_merge function."""
    dict_merged = defaultdict(list)
    for k, v in dict_.items():
        k_ = f_merge(k)
        dict_merged[k_] += v
    return dict_merged


def function_on_lists(lists, func):
    list_all = []
    for list_ in lists:
        list_all += list_
    return func(list_all)


def flatten(lists_or_tuples):
    list_all = []
    for x in lists_or_tuples:
        if not (isinstance(x, list) or isinstance(x, tuple)):
            list_all.append(x)
        # elif isflat(x):
        #     list_all.extend(x)
        else:
            list_all.extend(flatten(x))
    return list_all