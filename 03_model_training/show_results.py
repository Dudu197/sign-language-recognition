import sys
import json
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

if len(sys.argv) == 1:
    print("Usage: ref, dataset")
    exit(0)

ref = sys.argv[1]
dataset_name = sys.argv[2]
results_path = f"../99_model_output/results/{ref}/{dataset_name}/"

results = []
for file in os.listdir(results_path):
    if file.endswith(".json"):
        with open(os.path.join(results_path, file), "r") as f:
            results.append(json.load(f))

def get_metric(y_true, y_pred, metric, average):
    if metric == accuracy_score:
        return metric(y_true, y_pred)
    else:
        return metric(y_true, y_pred, average=average, zero_division=0)

def get_average_metrics(metric, average):
    values = []
    for r in results:
        y_true = r["true_labels"]
        y_pred = r["predicted_labels"]
        values.append(get_metric(y_true, y_pred, metric, average))
    return values

def print_metrics(metric, average):
    metric_results = get_average_metrics(metric, average)
    average = sum(metric_results) / len(metric_results)
    std = np.std(metric_results)
    print(f"{metric.__name__}:", average)

result = results[0]

print(f"{dataset_name} #{ref}")
print("epochs:", result["epochs"])
print("optimizer_parameters:", result["optimizer_parameters"])
print("image_method:", result.get("image_method", "Skeleton-DML"))
print("model:", result.get("model", "resnet18"))
print("meta:", result.get("meta"))
print()
print_metrics(accuracy_score, None)
print_metrics(precision_score, "macro")
print_metrics(recall_score, "macro")
print_metrics(f1_score, "macro")

print("=" * 10)
