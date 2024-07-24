#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[125]:


import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
from matplotlib import pyplot as plt
from lopo_dataset import LopoDataset
from image_representations.base_image_representation import BaseImageRepresentation
from models import BaseModel


# In[11]:

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_name", help="Name of the dataset", required=True)
parser.add_argument("-r", "--ref", help="Number of reference", required=True)
parser.add_argument("-s", "--seed", help="Seed to be used", default="1638102311")
parser.add_argument("-vp", "--validate_people", help="People in validate group")
parser.add_argument("-tp", "--test_people", help="People in test group", required=True)
parser.add_argument("-lr", "--learning_rate", help="Leaning rate", required=True)
parser.add_argument("-wd", "--weight_decay", help="Weight Decay", required=True)
parser.add_argument("-im", "--image_method", help="Image method name", required=True)
parser.add_argument("-m", "--model", help="Model name", required=True)

args = parser.parse_args()

print("="*10)
print("Running", args)
print(datetime.now())

dataset_name = args.dataset_name
image_method_name = args.image_method
model_name = args.model
ref = args.ref
seed = int(args.seed)

image_method_type = BaseImageRepresentation.get_by_name(image_method_name)
if image_method_type is None:
    raise ValueError(f"{image_method_name} image method not found")
image_method = image_method_type()

if not os.path.exists(os.path.join("../99_model_output/results", ref, dataset_name)):
    if not os.path.exists(os.path.join("../99_model_output/results", ref)):
        os.mkdir(os.path.join("../99_model_output/results", ref))
    os.mkdir(os.path.join("../99_model_output/results", ref, dataset_name))
    os.mkdir(os.path.join("../99_model_output/results", ref, dataset_name, "models"))


# In[13]:


torch.manual_seed(seed)


# In[14]:

if dataset_name == "minds":
    # df = pd.read_csv("../00_datasets/dataset_output/libras_minds/libras_minds_openpose_80_frames.csv")
    df = pd.read_csv("../00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv")
    frames = 80
elif dataset_name == "ufop":
    df = pd.read_csv("../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose.csv")
    # df = pd.read_csv("../00_datasets/dataset_output/libras_ufop/libras_ufop_openpose_60_frames.csv")
    frames = 60
elif dataset_name == "ksl":
    df = pd.read_csv("../00_datasets/dataset_output/KSL/ksl_openpose.csv")
    frames = 1
else:
    raise ValueError("Invalid dataset name")


# In[15]:


df


# In[16]:


# Minds only
if dataset_name == "minds":
    if "person" not in df.columns:
        import re
        df["person"] = df["video_name"].apply(lambda i: int(re.findall(r".*Sinalizador(\d+)-.+.mp4", i)[0]))

if dataset_name == "ksl":
    if "person" not in df.columns:
        df["person"] = df["video_name"].apply(lambda i: int(i.split("\\")[1].split("_")[0]))


# In[17]:


# frames = 80
people = df["person"].unique()
if args.validate_people:
    validate_people = [int(i) for i in args.validate_people.split(",")]
else:
    validate_people = []
test_people = [int(i) for i in args.test_people.split(",")]
# train_people = people[:-2]

epochs = 20


# In[18]:

# In[23]:


num_features = len(df["category"].unique())
num_features


# In[24]:


# Modify the fully connected layer to match the number of classes
# resnet.fc = nn.Linear(num_ftrs, num_features)


# In[25]:

base_model = BaseModel.get_by_name(model_name)(num_features)
model = base_model.get_model()


# Define image transformations
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.Resize(base_model.image_size),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[19]:


print("Processing train")
train_dataset = LopoDataset(df, frames, transform, transform_distance=False, augment=True, person_out=validate_people + test_people, seed=seed, image_method=image_method)
print("Processing test")
test_dataset = LopoDataset(df, frames, transform, transform_distance=False, augment=False, person_in=test_people, seed=seed, image_method=image_method)
if validate_people:
    print("Processing validate")
    validate_dataset = LopoDataset(df, frames, transform, transform_distance=False, augment=False, person_in=validate_people, seed=seed, image_method=image_method)
print("Data loaded")


# In[1313]:


# Load dataset
# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
# validate_dataset = datasets.ImageFolder(root=validate_dir, transform=transform)


# In[1314]:


# # Split dataset into train and test
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# In[20]:


# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
if validate_people:
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False)


# In[21]:


# Load pre-trained ResNet18 model
# model = resnet18(pretrained=True)


# In[22]:


# num_ftrs = model.fc.in_features

# Add an extra dense layer
# resnet.fc = nn.Sequential(
#     nn.BatchNorm1d(num_ftrs),
#     nn.Linear(num_ftrs, 128),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(128, num_features)
# )


# In[46]:


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(resnet.parameters(), lr=10e-5)
optimizer_parameters = {
    "lr": float(args.learning_rate),
    "weight_decay": float(args.weight_decay)
}
optimizer = optim.Adam(model.parameters(), **optimizer_parameters)


# In[27]:


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[30]:


len(test_loader.dataset)


# In[31]:


history = {"loss": [], "accuracy": [], "val_accuracy": []}
best_val_loss = float('inf')
best_val_accuracy = 0
best_model_weights = model.state_dict()
patience = 5
counter = 0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        if len(inputs) == 1:
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train / total_train
    history["loss"].append(float(epoch_loss))
    history["accuracy"].append(float(train_accuracy))

    if validate_people:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_accuracy = correct / total
            val_loss = criterion(outputs, labels)
            history["val_accuracy"].append(float(val_accuracy))
    else:
        val_accuracy = train_accuracy
        val_loss = epoch_loss

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered. No improvement in validation loss.")
            break

    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(str(datetime.now()))


# In[32]:


# Load the best model weights
model.load_state_dict(best_model_weights)
print("Best val accuracy:", best_val_accuracy)


# In[33]:


# plt.plot(history["loss"])


# In[34]:


model.eval()
correct = 0
total = 0
class_correct = list(0. for _ in range(num_features))
class_total = list(0. for _ in range(num_features))


# In[35]:


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


# In[36]:


# Evaluate on test set
model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = correct / total
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')


# In[37]:


print(f"Accuracy on the test set: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# In[38]:


accuracy = correct / total
print(f"Accuracy on the test set: {accuracy:.4f}")


# In[39]:


categories = [i+1 for i in range(num_features)]


# In[40]:


precisions = []
for i in range(num_features):
    if class_total[i] > 0:
        precision = class_correct[i] / class_total[i]
    else:
        precision = -1
    precisions.append(precision)
    if precision < 1:
        print(f"Precision of class {i}: {precision:.4f}")
# plt.bar(categories, precision)


# In[41]:


# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# plt.figure(figsize=(12, 10))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()


# In[42]:


0.91 - accuracy


# In[43]:


# plt.plot(history["loss"])


# In[44]:


# plt.plot(history["val_accuracy"])


# In[121]:


result = {
    "dataset_name": dataset_name,
    "frames": frames,
    "seed": seed,
    "epochs": epochs,
    "last_epoch": epoch,
    "optimizer_parameters": optimizer_parameters,
    "test_people": test_people,
    "validate_people": validate_people,
    # "resnet_fc_layer": str(model.fc),
    "history": history,
    "true_labels": true_labels,
    "predicted_labels": predicted_labels,
    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": float(recall),
    "test_f1_score": float(f1),
    "precision_per_test_class": precisions,
    "image_method": image_method_name,
    "model": base_model.name,
}


# In[122]:



# In[123]:


# In[124]:


file_name = str(datetime.now()).replace(" ", "_")
file_path = os.path.join("../99_model_output/results", ref, dataset_name, file_name + ".json")
model_path = os.path.join("../99_model_output/results", ref, dataset_name, "models", file_name + ".pth")
with open(file_path, "w") as f:
    json.dump(result, f, default=int)

torch.save(model.state_dict(), model_path)

# In[ ]:




