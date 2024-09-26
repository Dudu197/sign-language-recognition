# Sign Language Recognition

This repository contains the code for the paper "Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation".

# Disclaimer

This repository is under refactoring. The code was original design for experiments, and now we are working to make it more user-friendly.

If you only want to train the model, you can use the [sign-language-recognition-model](https://github.com/Dudu197/sign-language-recognition-model) repository.

It is a simpler version of the original code, where you can run in a few minutes.

If you have any questions, feel free to open an issue.

# Content

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)

# Introduction

Sign Language Recognition (SLR) is a challenging task that has been widely studied in the literature.
In this work, we propose a novel approach to SLR based on skeleton images.

# Requirements

The code was implemented in Python 3.7 and the following libraries are required:


# Dataset

We used two Brazilian Sign Language (Libras) datasets to evaluate our model: MINDS-Libras and Libras-UFOP.

## MINDS-Libras

The [MINDS-Libras dataset](https://link.springer.com/article/10.1007/s00521-021-05802-4) consists in 20 signs, with 12 signers performing the sign 5 times.

The dataset is publicly available at [Zenodo](https://zenodo.org/records/2667329)

## Libras-UFOP

The [Libras-UFOP dataset](https://www.sciencedirect.com/science/article/pii/S0957417420309143) has 56 signs, performed by 5 signers, repeating each sign from 8 to 16 times.

# Preprocessing

The preprocessing step is responsible for transforming the dataset into landmark points.

The folder `01_landmark_extraction` contains the code for extracting the landmark points from videos into multiple CSV using OpenPose.

The folder `02_data_processing` contains the code for join the CSV files into a single file.

# Training

The training step is responsible for training the model using the dataset and hyperparameters.

The folder `03_model_training` contains the code for training the model.

Most of the hyperparameters are already parallelized, but some of them need to be adjusted according to the dataset.

You can see more details about in the `03_model_training/README.md` file.


# Results

Our model was able to overcome the state-of-the-art on both MINDS-Libras and Libras-UFOP datasets.

On MINDS-Libras, the model had an accuracy of 0.93, we had an improvement of 2 percentage points on accuracy and 3 percentage points on F1-Score, comparing to the state-of-the-art.

For Libras-UFOP dataset, the difference is ever bigger. Our model shows an accuracy of 0.82, 8 percentage points above the state-of-the-art and 9 percentage points on F1-score.


# Citation

If you use this code for your research, please cite our paper:

Citation:
```
Alves, Carlos Eduardo GR, Francisco de Assis Boldt, and Thiago M. Paixão. "Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation." arXiv preprint arXiv:2404.19148 (2024). 
```

Bibtex:
```
@article{alves2024enhancing,
  title={Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation},
  author={Alves, Carlos Eduardo GR and Boldt, Francisco de Assis and Paix{\~a}o, Thiago M},
  journal={arXiv preprint arXiv:2404.19148},
  year={2024}
}
```

