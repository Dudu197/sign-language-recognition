import pandas as pd
from datetime import datetime
from model_training import ModelTraining

model_training = ModelTraining()
print("Loading dataset")
print(datetime.now())

dataset_name = model_training.args.dataset_name

if dataset_name == "ksl":
    df = pd.read_csv("../00_datasets/dataset_output/KSL/ksl_openpose.csv")
else:
    raise ValueError("Invalid dataset name")


if dataset_name == "ksl":
    if "person" not in df.columns:
        df["person"] = df["video_name"].apply(lambda i: int(i.split("\\")[1].split("_")[0]))

epochs = 50

model_training.train(df, epochs)
