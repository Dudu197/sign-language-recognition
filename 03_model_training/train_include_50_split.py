import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from model_training import ModelTraining

model_training = ModelTraining()
print("Loading dataset")
print(datetime.now())

dataset_name = model_training.args.dataset_name

# In[14]:

df = pd.read_csv("../00_datasets/dataset_output/include50/include50_openpose.csv")
df["video_path"] = df.apply(lambda i: (i["video_name"].replace("\\", "/")).lower().replace(".mov", "").replace(".mp4", ""), axis=1)

base_path = "../00_datasets/dataset_output/include50/"
train_file = "include50_train.txt"
validate_file = "include50_val.txt"
test_file = "include50_test.txt"

def read_file(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines

with open(base_path + train_file) as file:
    train_videos = [line.rstrip().lower().replace(".mov", "").replace(".mp4", "") for line in file]

with open(base_path + test_file) as file:
    test_videos = [line.rstrip().lower().replace(".mov", "").replace(".mp4", "") for line in file]

with open(base_path + validate_file) as file:
    validate_videos = [line.rstrip().lower().replace(".mov", "").replace(".mp4", "") for line in file]


# train_df = df[df["video_path"].isin(train_videos)]
# test_df = df[df["video_path"].isin(test_videos)]
# validate_df = df[df["video_path"].isin(validate_videos)]

df["person"] = -1
df.loc[df["video_path"].isin(train_videos), "person"] = 0
df.loc[df["video_path"].isin(validate_videos), "person"] = 1
df.loc[df["video_path"].isin(test_videos), "person"] = 2

df = df[df["person"] != -1]

epochs = 50

model_training.train(df, epochs)
