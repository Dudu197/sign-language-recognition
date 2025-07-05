# Sign Language Recognition

Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation

---

## Overview

This repository contains the code for the paper:

> **Alves, Carlos Eduardo GR, Francisco de Assis Boldt, and Thiago M. Paixão. "Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation." arXiv preprint arXiv:2404.19148 (2024).**

We propose a novel approach to Sign Language Recognition (SLR) using skeleton image representations, achieving state-of-the-art results on Brazilian Sign Language datasets.

---

## Table of Contents
- [Overview](#overview)
- [Disclaimer](#disclaimer)
- [Requirements & Installation](#requirements--installation)
- [Dataset](#dataset)
- [Directory Structure](#directory-structure)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Disclaimer

This repository is under refactoring. The code was originally designed for experiments and is being improved for user-friendliness.

If you **only want to train the model**, use the [sign-language-recognition-model](https://github.com/Dudu197/sign-language-recognition-model) repository for a simpler, quick-start version.

For questions, please open an issue.

---

## Requirements & Installation

- **Python 3.7**
- Required libraries are listed in `requirements.txt`.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Dataset

We use two Brazilian Sign Language (Libras) datasets:

### MINDS-Libras
- 20 signs, 12 signers, 5 repetitions per sign.
- [Dataset on Zenodo](https://zenodo.org/records/2667329)
- [Paper](https://link.springer.com/article/10.1007/s00521-021-05802-4)

### Libras-UFOP
- 56 signs, 5 signers, 8–16 repetitions per sign.
- [Paper](https://www.sciencedirect.com/science/article/pii/S0957417420309143)

### Preprocessed Datasets

For convenience, we provide preprocessed versions of the datasets with extracted landmarks:

- **MINDS-Libras:** [Download preprocessed data](https://drive.google.com/file/d/1qx2JudpjPgpp4-fpJ7YVMrszWV4lYPCd/view?usp=drive_link)
  - Place at: `00_datasets/dataset_output/libras_minds/libras_minds_openpose.csv`

- **Include-50:** [Download preprocessed data](https://drive.google.com/file/d/14SbYpFIbHi_Is1hD9XH9Sg_5eF--xtAw/view?usp=sharing)
  - Place at: `00_datasets/dataset_output/include50/include50_openpose.csv`

- **KSL (Korean Sign Language):** [Download preprocessed data](https://drive.google.com/file/d/1-27qX-KtCE3RknzASvXuJ-aVZAQ60tNj/view?usp=sharing)
  - Place at: `00_datasets/dataset_output/KSL/ksl_openpose.csv`

---

## Directory Structure

- `01_landmarks_extraction/` – Extract landmark points from videos using OpenPose.
- `02_data_processing/` – Join CSV files into a single dataset.
- `03_model_training/` – Model training scripts and model definitions.
- `04_result_analysis/` – Notebooks and scripts for analyzing results.
- `00_data_exploration/` – Data exploration scripts and notebooks.
- `99_model_output/` – Model outputs.
- `99_old/` – Legacy scripts and notebooks.
- `99_others/` – Miscellaneous scripts and data.
- `99_skeleton_explore/` – Skeleton-based experiments.

---

## Preprocessing

1. **Extract Landmarks:**
   - Use scripts in `01_landmarks_extraction/` to extract landmark points from videos (e.g., with OpenPose).
2. **Join CSVs:**
   - Use scripts in `02_data_processing/` to merge CSVs into a single dataset file.

---

## Training

- Training scripts are in `03_model_training/`.
- Most hyperparameters are parallelized, but some may need adjustment per dataset.
- See `03_model_training/README.md` for details.

**Example:**
```bash
python 03_model_training/model_training.py --config your_config.yaml
```

---

## Results

Our model achieves strong performance across multiple sign language datasets:

### Brazilian Sign Language Datasets
- **MINDS-Libras:**
  - Accuracy: **0.93**
  - +2 percentage points accuracy, +3 F1-Score over previous SOTA
- **Libras-UFOP:**
  - Accuracy: **0.82**
  - +8 percentage points accuracy, +9 F1-Score over previous SOTA

### Additional Datasets
- **Include-50:**
  - Accuracy: **0.97** (ResNet18 + Skeleton-DML)
  - Excellent performance on this larger dataset
- **KSL (Korean Sign Language):**
  - Accuracy: **0.63** (ResNet18 + Skeleton-DML)
  - Best performance with Skeleton-DML representation

---

## Citation

If you use this code for your research, please cite our paper:

> Alves, Carlos Eduardo GR, Francisco de Assis Boldt, and Thiago M. Paixão. "Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation." arXiv preprint arXiv:2404.19148 (2024).

**BibTeX:**
```bibtex
@article{alves2024enhancing,
  title={Enhancing Brazilian Sign Language Recognition through Skeleton Image Representation},
  author={Alves, Carlos Eduardo GR and Boldt, Francisco de Assis and Paix{\~a}o, Thiago M},
  journal={arXiv preprint arXiv:2404.19148},
  year={2024}
}
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, bug fixes, or improvements.

---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

