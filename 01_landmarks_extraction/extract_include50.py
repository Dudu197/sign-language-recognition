import openpose_extraction as openpose_extraction
import os

base_path = "D:\\Projects\\datasets\\include50"
output_path = "../00_datasets/dataset_output/include50/raw"

processed = os.listdir(output_path)
# all_videos = []

for base_category in os.listdir(base_path):
    videos_to_process = []
    if os.path.isfile(os.path.join(base_path, base_category)):
        print(f"Skipping file {base_category}")
        continue
    if any([base_category in processed_category for processed_category in processed]):
        print(f"Skipping category {base_category}")
        continue
    for category in os.listdir(os.path.join(base_path, base_category)):
        for video in os.listdir(os.path.join(base_path, base_category, category)):
            # if video == "Extra":
            #     print("Extra!!!!!!")
            #     for extra_video in os.listdir(os.path.join(base_path, base_category, category, video)):
            #         video_path = os.path.join(base_path, base_category, category, video, extra_video)
            #         video_name = os.path.join(base_category, category, video, extra_video)
            #         signaler = 0
            #         videos_to_process.append((video_path, video_name, category, signaler, signaler))
            video_path = os.path.join(base_path, base_category, category, video)
            video_name = os.path.join(base_category, category, video)
            signaler = 0
            videos_to_process.append((video_path, video_name, category, signaler, signaler))
    print(f"Processing {base_category}")
    # all_videos = all_videos + videos_to_process
    df = openpose_extraction.process(videos_to_process)
    df.to_csv(os.path.join(output_path, f"include50_{base_category}.csv"))
# df.to_csv(os.path.join(output_path, f"include50_extra.csv"))
