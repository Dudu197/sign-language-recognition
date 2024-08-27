#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys


# In[2]:


# df = pd.read_csv("../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose.csv")
df = pd.read_csv("../00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv")




# In[8]:


mean = 65
[i * mean for i in [0.8, 0.9, 1.1, 1.2]]


# In[9]:


target_frames = int(sys.argv[1])
print(target_frames)


# In[10]:


df_new_fps = pd.DataFrame(columns=df.columns)
for video_name in df["video_name"].unique():
    df_video = df[df["video_name"] == video_name]
    df_video = df_video[df_video["frame"] %2 == 0]
    video_frames = len(df_video)
    df_video["frame"] = [i for i in range(video_frames)]
    df_new_fps = pd.concat([df_new_fps, df_video])

df = df_new_fps

df


# In[11]:


# df_video = df[df["video_name"] == "001_001_001.mp4"]
# video_frames = df_video["frame"].max()
# print(video_frames)
# frames_diff = video_frames - target_frames
# print(frames_diff)
# frames_to_remove = [i for i in range(0, video_frames, video_frames//(video_frames-target_frames))]
# # frames_to_remove = [video_frames]
# frames_to_remove = frames_to_remove[(len(frames_to_remove) - frames_diff - 1):]
# # if len(frames_to_remove) < frames_diff:
# #     frames_to_remove += [i for i in range(0, len(frames_to_remove) - frames_diff)]
# print(frames_to_remove)
# new_df = df_video[~df_video["frame"].isin(frames_to_remove)]
# new_df


# In[12]:


df.groupby("video_name").max()["frame"].hist()


# In[13]:


df.groupby("video_name").max()["frame"].mean()


# In[14]:


df["category"].unique()


# In[15]:


df.groupby("video_name")["frame"].max()


# In[16]:


fig, ax = plt.subplots()
fig.set_dpi(200)

count, n_bins, patches = plt.hist(df.groupby("video_name")["frame"].max(), bins=int((240-60)/20))
# plt.xticks(range(60, 240, 20))
i = -1
for count, n_bin, patch in zip(count, n_bins, patches):
    i += 1
    # Get the center of the bar
    x = patch.get_x() + patch.get_width() / 2
    # Add the text annotation
    plt.text(x, count, str(f"{int(n_bins[i])} - {int(n_bins[i + 1]) - 1}"), ha='center', va='bottom')
    plt.text(x, count + 15, str(int(count)), ha='center', va='bottom')



plt.xticks(n_bins)
ax.grid(axis='y')
plt.ylabel("Video Samples")
plt.xlabel("Frame Count")
# plt.title("Tamanho do vídeo em frames")


# In[17]:


n_bins


# In[18]:


df.groupby("video_name")["frame"].max().min()


# In[19]:


# 140, 160


# In[20]:


df_new_fps = pd.DataFrame(columns=df.columns)
for video_name in df["video_name"].unique():
    df_video = df[df["video_name"] == video_name]
    video_frames = len(df_video)
    if video_frames == target_frames:
        new_df = df_video
    else:
        if video_frames < target_frames:
            new_df = df_video
            last_frame = df_video[df_video["frame"] == video_frames - 1]
            frames_count = target_frames - video_frames
            new_df = pd.concat([last_frame] * frames_count)
            new_df["frame"] = np.arange(video_frames, target_frames)
            new_df = pd.concat([df_video, new_df])
        else:
            frames_diff = video_frames - target_frames
            frames_to_remove = [i for i in range(0, video_frames, video_frames//(video_frames-target_frames))]
            if len(frames_to_remove) > frames_diff:
                frames_to_remove = frames_to_remove[:frames_diff]
            if len(frames_to_remove) < frames_diff:
                print("frames diff menor")
            new_df = df_video[~df_video["frame"].isin(frames_to_remove)]
            if len(new_df) > target_frames:
                new_df = new_df.iloc[:target_frames]
            if len(new_df) < target_frames:
                print("menor")
                break
    df_new_fps = pd.concat([df_new_fps, new_df])


# In[21]:


df_new_fps


# In[22]:


df_new_fps["category"].value_counts()


# In[23]:


df_new_fps["category"].unique()


# In[24]:


len(df_new_fps["category"].unique())


# In[25]:


grouped_count = df_new_fps.groupby("video_name")["frame"].count()


# In[26]:


grouped_count[grouped_count > target_frames]


# In[27]:


grouped_count[grouped_count < target_frames]


# In[28]:


video_names = list(grouped_count[grouped_count > target_frames].index)
for video_name in video_names:
    df_video = df_new_fps[df_new_fps["video_name"] == video_name]
    extra_frames = len(df_video) - target_frames
    frames_to_remove = df_video.iloc[-extra_frames:]
    df_new_fps = df_new_fps.drop(frames_to_remove.index)


# In[29]:


len(df_new_fps.video_name.unique()) * target_frames


# In[30]:


df_new_fps.groupby("video_name")["frame"].count()


# In[31]:


df_new_fps.to_csv(f"../00_datasets/dataset_output/libras_minds/libras_minds_openpose_{target_frames}_frames_sample.csv", index=False)
# df_new_fps.to_csv(f"../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose_{target_frames}_frames.csv", index=False)


# In[ ]:




