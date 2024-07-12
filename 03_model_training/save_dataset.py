import numpy as np

from lopo_dataset import LopoDataset
import pandas as pd
import re


df = pd.read_csv("../00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv")
df["person"] = df["video_name"].apply(lambda i: int(re.findall(r".*Sinalizador(\d+)-.+.mp4", i)[0]))
dataset = LopoDataset(df, 80, None, transform_distance=False, augment=False, seed=101)
df_1 = dataset.df.iloc[0]
x = df_1["x"]
y = df_1["y"]
z = df_1["z"]

with open("image_representations/data.npy", "wb") as f:
    np.save(f, x)
    np.save(f, y)
    np.save(f, z)
# df_1.to_csv("data.csv")
