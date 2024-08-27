import pandas as pd
from datetime import datetime
from model_training import ModelTraining

model_training = ModelTraining()
print("Loading dataset")
print(datetime.now())

dataset_name = model_training.args.dataset_name
ref = model_training.args.ref


if dataset_name == "minds":
    # [110.4, 124.2, 138, 151.8, 165.6]
    # [55.2, 62.1, 69, 75.9, 82.8]
    # ref_frames = {
    #     "49": 55,
    #     "50": 62,
    #     "51": 69,
    #     "52": 76,
    #     "53": 83,
    # }
    # frames = ref_frames[ref]
    # print(f"Frames: {frames}")
    # df = pd.read_csv(f"../00_datasets/dataset_output/libras_minds/libras_minds_openpose_{frames}_frames_sample.csv")
    df = pd.read_csv("../00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv")
elif dataset_name == "ufop":
    df = pd.read_csv("../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose.csv")
    # [52.0, 58.5, 71.5, 78.0]
    # ref_frames = {
    #     "44": 52,
    #     "45": 59,
    #     "46": 65,
    #     "47": 71,
    #     "48": 78,
    # }
    # frames = ref_frames[ref]
    # print(f"Frames: {frames}")
    # df = pd.read_csv(f"../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose_{frames}_frames.csv")
elif dataset_name == "ksl":
    df = pd.read_csv("../00_datasets/dataset_output/KSL/ksl_openpose.csv")
    frames = 1
else:
    raise ValueError("Invalid dataset name")

# Minds only
if dataset_name == "minds":
    if "person" not in df.columns:
        import re
        df["person"] = df["video_name"].apply(lambda i: int(re.findall(r".*Sinalizador(\d+)-.+.mp4", i)[0]))

if dataset_name == "ksl":
    if "person" not in df.columns:
        df["person"] = df["video_name"].apply(lambda i: int(i.split("\\")[1].split("_")[0]))

epochs = 20

model_training.train(df, epochs)
