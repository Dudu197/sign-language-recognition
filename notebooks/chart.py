from collections import defaultdict
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import load_rgb
from skimage import exposure, img_as_float

DPI = 10


def show(
    image,
    ax=None,
    rotate=False,
    scale=1,
    title=None,
    fontsize=10,
    cmap=None,
    return_ax=False,
):
    h, w = image.shape[:2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(int(scale * h), int(scale * w)), dpi=DPI)
    ax.cla()
    ax.axis("off")
    if rotate:
        param = (1, 0) if image.ndim == 2 else (1, 0, 2)
        image = np.transpose(image, param)[::-1]
    ax.imshow(image, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=DPI * fontsize)
    if return_ax:
        return ax


def show_collection(
    images,
    titles=[],
    num_rows=-1,
    num_cols=-1,
    scale=1,
    cmap=None,
    return_axes=False,
    fontsize=10,
    # pad=1.0,
):
    assert len(images) > 1

    if num_cols == -1:
        # none provided: fix row in 1
        if num_rows == -1:
            num_rows = 1
        # compute #cols based on #rows
        num_cols = len(images) // num_rows
    else:
        # only #cols provided: fix cols, compute rows
        if num_rows == -1:
            num_rows = math.ceil(len(images) / num_cols)
        # both rows and cols provided: fix rows, compute cols
        else:
            num_cols = math.ceil(len(images) / num_rows)

    if len(titles) > 0:
        assert len(titles) == len(images)
    else:
        titles = len(images) * [""]

    h, w = images[0].shape[:-1] if images[0].ndim == 3 else images[0].shape
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(int(scale * w * num_cols), int(scale * h * num_rows)),
        dpi=DPI,
    )
    k = 1
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image, cmap=cmap)
        ax.axis("off")
        ax.set_title(title, fontsize=DPI * fontsize)
        k += 1
    fig.tight_layout()
    if return_axes:
        return axes


# def draw_bbox(ax, bbox, color=(1.0, 0, 1.0)):
#     rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor=color, facecolor="none")
#     ax.add_artist(rect)
#     return ax


def gray2rgb(image):
    return np.transpose(np.stack(3 * [image]), (1, 2, 0))


def list2table(mlist, n=20):
    line = []
    lines = [" | ".join(n * ["[]()"]), "|".join(n * ["-----"])]
    k = 0
    for w in mlist:
        line.append(w)
        if (k + 1) % n == 0:
            lines.append(" | ".join(line))
            line = []
        k += 1
    sep = "\n{}\n".format("|".join(n * ["-----"]))
    text = "\n".join(lines)
    return text


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height.
    https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{:.2f}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_histogram(images, titles=None, bins=256, return_as_image=False, fontsize=10):
    num_images = 1
    if isinstance(images, list):
        num_images = len(images)
    else:
        images = [images]

    fig, axes = plt.subplots(ncols=num_images, sharey=True, figsize=(num_images * 3, 3))
    if num_images == 1:
        axes = np.array(axes)
    axes = axes.ravel()
    axes_twins = []
    if titles is None:
        titles = num_images * [""]

    for ax, image, title in zip(axes, images, titles):
        image = img_as_float(image)
        ax.hist(image.ravel(), bins=bins, histtype="step", color="black")
        ax.set_title(title, fontsize=fontsize)
        ax_cdf = ax.twinx()
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, "r")
        axes_twins.append(ax_cdf)

    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    plt.xlabel("Pixel intensity", fontsize=fontsize)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylabel("Number of pixels", fontsize=fontsize)
    _, y_max = axes[0].get_ylim()
    axes[0].set_yticks(np.linspace(0, y_max, 5))

    axes_twins[-1].set_yticks(np.linspace(0, 1, 5))
    axes_twins[-1].set_ylabel("Fraction of total intensity", fontsize=fontsize)

    if return_as_image:
        plt.close()
        fig.canvas.draw()  # cache the renderer
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        fig_image = data.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)
        return fig_image


def plot_dist(values, topn=30):
    fig, ax = plt.subplots(figsize=(15, 3))
    count = defaultdict(lambda: 0)
    for val in values:
        count[val] += 1
    count_sorted = sorted(count.items(), key=lambda item: item[1], reverse=True)[:topn]
    labels, values = zip(*count_sorted)
    # labels
    plt.bar(range(len(count_sorted)), values, align="center")
    plt.xticks(range(len(count_sorted)), labels, fontsize=12)
    plt.yticks(fontsize=12)


def put_frame(image, is_rgb=False, thickness=1, color=(255, 0, 0)):
    h, w = image.shape[:2]
    bbox = (0, 0, w, h)
    image_frame = draw_bbox(image, bbox, is_rgb, thickness, color=color)
    return image_frame


# def draw_bbox(image, bbox, is_rgb=False, thickness=1, color=(255, 0, 0)):
#     if image.dtype in [np.bool, np.float32]:
#         image = (255 * image).astype(np.uint8)
#     if not is_rgb:
#         image = np.stack([image, image, image], axis=0).transpose([1, 2, 0])

#     x, y, w, h = bbox
#     x1 = x
#     y1 = y
#     x2 = x1 + w - 1
#     y2 = y1 + h - 1
#     image = image.copy()
#     image_rec = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#     return image_rec
