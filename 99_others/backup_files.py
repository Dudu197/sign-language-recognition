import os
import shutil

base_path = "results"
output_path = "results_copy"


def mkdir(path):
    path = path.replace(base_path, output_path)
    if not os.path.exists(path):
        os.mkdir(path)
        # print(f"Creating folder {path}")


def cp(src):
    shutil.copy2(src, src.replace(base_path, output_path).replace('', '_'))
    # print(f"Copying file {src.replace('', '_')}")


for execution in os.listdir(base_path):
    path = os.path.join(base_path, execution)
    mkdir(path)
    for run in os.listdir(path):
        run_path = os.path.join(path, run)
        if run.endswith(".txt"):
            cp(run_path)
            continue
        mkdir(run_path)
        for dataset in os.listdir(run_path):
            dataset_path = os.path.join(run_path, dataset)
            if dataset.endswith(".txt") or dataset.endswith(".json"):
                cp(dataset_path)
