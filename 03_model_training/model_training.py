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


class ModelTraining:

    def __init__(self):
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

        self.args = parser.parse_args()

    def train(self, df, epochs, meta=None):
        print("=" * 10)
        print("Running", self.args)
        print(datetime.now())

        dataset_name = self.args.dataset_name
        image_method_name = self.args.image_method
        model_name = self.args.model
        ref = self.args.ref
        seed = int(self.args.seed)

        image_method_type = BaseImageRepresentation.get_by_name(image_method_name)
        if image_method_type is None:
            raise ValueError(f"{image_method_name} image method not found")
        image_method = image_method_type()

        if not os.path.exists(os.path.join("../99_model_output/results", ref, dataset_name)):
            if not os.path.exists(os.path.join("../99_model_output/results", ref)):
                os.mkdir(os.path.join("../99_model_output/results", ref))
            os.mkdir(os.path.join("../99_model_output/results", ref, dataset_name))
            os.mkdir(os.path.join("../99_model_output/results", ref, dataset_name, "models"))


        torch.manual_seed(seed)

        # frames = 80
        # people = df["person"].unique()
        if self.args.validate_people and self.args.validate_people != "-1":
            validate_people = [int(i) for i in self.args.validate_people.split(",")]
        else:
            validate_people = []
        test_people = [int(i) for i in self.args.test_people.split(",")]
        # train_people = people[:-2]


        num_features = len(df["category"].unique())
        num_features

        base_model = BaseModel.get_by_name(model_name)(num_features)
        model = base_model.get_model()

        # Define image transformations
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Resize(base_model.image_size),
            transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Processing train")
        train_dataset = LopoDataset(df, transform, transform_distance=False, augment=True,
                                    person_out=validate_people + test_people, seed=seed, image_method=image_method)
        print("Processing test")
        test_dataset = LopoDataset(df, transform, transform_distance=False, augment=False, person_in=test_people,
                                   seed=seed, image_method=image_method)
        if validate_people:
            print("Processing validate")
            validate_dataset = LopoDataset(df, transform, transform_distance=False, augment=False,
                                           person_in=validate_people, seed=seed, image_method=image_method)
        print("Data loaded")

        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        if validate_people:
            validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer_parameters = {
            "lr": float(self.args.learning_rate),
            "weight_decay": float(self.args.weight_decay)
        }
        optimizer = optim.Adam(model.parameters(), **optimizer_parameters)

        # Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        len(test_loader.dataset)


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

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(str(datetime.now()))

        # Load the best model weights
        model.load_state_dict(best_model_weights)
        print("Best val accuracy:", best_val_accuracy)

        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for _ in range(num_features))
        class_total = list(0. for _ in range(num_features))

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


        print(f"Accuracy on the test set: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        accuracy = correct / total
        print(f"Accuracy on the test set: {accuracy:.4f}")

        categories = [i + 1 for i in range(num_features)]


        precisions = []
        for i in range(num_features):
            if class_total[i] > 0:
                precision = class_correct[i] / class_total[i]
            else:
                precision = -1
            precisions.append(precision)
            if precision < 1:
                print(f"Precision of class {i}: {precision:.4f}")

        result = {
            "dataset_name": dataset_name,
            "seed": seed,
            "epochs": epochs,
            "last_epoch": epoch,
            "optimizer_parameters": optimizer_parameters,
            "test_people": test_people,
            "validate_people": validate_people,
            "resnet_fc_layer": str(base_model.get_fc_layer()),
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
            "meta": meta
        }

        file_name = str(datetime.now()).replace(" ", "_")
        file_path = os.path.join("../99_model_output/results", ref, dataset_name, file_name + ".json")
        model_path = os.path.join("../99_model_output/results", ref, dataset_name, "models", file_name + ".pth")
        with open(file_path, "w") as f:
            json.dump(result, f, default=int)

        torch.save(model.state_dict(), model_path)

