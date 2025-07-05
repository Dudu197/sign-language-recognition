import openpose_extraction as openpose_extraction
import os

base_path = "D:\\Projects\\datasets\\videos UFPE (V-LIBRASIL)\\data"
output_path = "../00_datasets/dataset_output/vlibras/raw"

processed = os.listdir(output_path)

categories = [i[0] for i in os.listdir(base_path)]

for category in categories:
    videos_to_process = []
    if f"vlibras_{category}.csv" in processed:
        print(f"Skipping category {category}")
        continue
    for video in os.listdir(os.path.join(base_path)):
        if video[0] != category:
            continue
        video_path = os.path.join(base_path, video)
        video_name = video
        video_category = video.split("_")[0]
        signaler = video[-5]
        videos_to_process.append((video_path, video_name, video_category, signaler, signaler))
    print(f"Processing {category}")
    # all_videos = all_videos + videos_to_process
    df = openpose_extraction.process(videos_to_process)
    df.to_csv(os.path.join(output_path, f"vlibras_{category}.csv"))
