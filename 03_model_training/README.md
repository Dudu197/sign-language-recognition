# Model Training

This folder contains all scripts and resources for training sign language recognition models using various datasets and architectures.

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Available Models & Image Representations](#available-models--image-representations)
- [Hyperparameters & Command-Line Arguments](#hyperparameters--command-line-arguments)
- [Datasets](#datasets)
- [Running Experiments](#running-experiments)
  - [Batch Training (LOPO)](#batch-training-lopo)
  - [Single Training](#single-training)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [References](#references)

## Overview
This module enables training and evaluation of deep learning models for sign language recognition. It supports different datasets, preprocessing methods, and model architectures. The code is designed for flexibility and reproducibility, supporting both single-run and batch (LOPO) experiments.

## Folder Structure
- `model_training.py` — Main script for training models.
- `lopo_dataset.py` — Utilities for Leave-One-Person-Out (LOPO) dataset splitting.
- `models/` — Model architecture definitions (e.g., ResNet, EfficientNet, ViT).
- `image_representations/` — Methods for converting skeleton data to image representations.
- `notebooks/` — Jupyter notebooks for exploration and prototyping.
- `run_all_batches.sh` — Script to automate batch LOPO training.
- `save_dataset.py` — Utility for saving processed datasets.
- `show_results.py` — Script for visualizing results.

## Available Models & Image Representations
- **Models:**
  - `resnet18`, `resnet50`, `efficientnet_b6`, `mobilenet_v4_hybrid_medium`, `vit_l_16`, `vit_medium`
- **Image Representations:**
  - `Skeleton-DML`, `Skeleton-Magnitude`, `SL-DML` (see `image_representations/` for details)

## Hyperparameters & Command-Line Arguments
The main training script (`model_training.py`) accepts the following arguments:

- `-d`, `--dataset_name`         — Dataset name (`minds`, `ufop`)
- `-s`, `--seed`                 — Random seed (int)
- `-vp`, `--validate_people`     — Validation people (comma-separated ints, e.g., `1,2,3`)
- `-tp`, `--test_people`         — Test people (comma-separated ints, e.g., `4,5`)
- `-lr`, `--learning_rate`       — Learning rate (float)
- `-wd`, `--weight_decay`        — Weight decay (float)
- `-im`, `--image_method`        — Image representation method (see above)
- `-m`, `--model`                — Model architecture (see above)
- `-r`, `--ref`                  — Reference number for experiment tracking (int)

## Pre-proccesed Datasets
- **MINDS-Libras:** Preprocessed version available [here](https://drive.google.com/file/d/1qx2JudpjPgpp4-fpJ7YVMrszWV4lYPCd/view?usp=drive_link).
  - Place at: `00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv`
- **Include-50:** Preprocessed version available [here](https://drive.google.com/file/d/14SbYpFIbHi_Is1hD9XH9Sg_5eF--xtAw/view?usp=sharing).
  - Place at: `00_datasets/dataset_output/include50/include50_openpose.csv`
- **KSL:** Preprocessed version available [here](https://drive.google.com/file/d/1-27qX-KtCE3RknzASvXuJ-aVZAQ60tNj/view?usp=sharing).
  - Place at: `00_datasets/dataset_output/KSL/ksl_openpose.csv`

## Running Experiments

### Batch Training (LOPO)
For Leave-One-Person-Out (LOPO) experiments, use the provided batch script:

```bash
./run_all_batches.sh
```
This will iterate over all splits, training and evaluating the model for each.

### Single Training
To run a single experiment with custom parameters:

```bash
python model_training.py -d minds -s 42 -vp 1,2,3 -tp 4,5 -lr 0.001 -wd 0.0001 -im Skeleton-DML -m resnet18 -r 1
```
