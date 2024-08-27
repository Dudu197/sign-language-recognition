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
df_videos = df[df["frame"] == 1]

pre_train_videos, test_videos = train_test_split(df_videos["video_name"], test_size=0.2, stratify=df_videos["category"])

pre_train_df = df_videos[df_videos["video_name"].isin(pre_train_videos)]
test_df = df[df["video_name"].isin(test_videos)]

train_videos, validate_videos = train_test_split(pre_train_df["video_name"], test_size=0.2, stratify=pre_train_df["category"])
train_df = df[df["video_name"].isin(train_videos)]
validate_df = df[df["video_name"].isin(validate_videos)]

df["person"] = 0
df.loc[df["video_name"].isin(validate_videos), "person"] = 1

epochs = 50

model_training.train(df, epochs)
