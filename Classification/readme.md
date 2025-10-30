
# 😴 Drowsiness Detection with PyTorch & ResNet-18

![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Language](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Model](https://img.shields.io/badge/Model-ResNet18-green.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen.svg)

This repository contains a PyTorch solution for a binary image classification task: **detecting drowsiness**.

The project uses **transfer learning** by leveraging a pretrained **ResNet-18** model, freezing its weights, and training only a new classifier head. The model is trained on a custom dataset where drowsiness is determined by the image's filename.

---

## 💾 The Dataset

The dataset (from `/kaggle/input/drowsy`) consists of thousands of images, categorized into two classes based on their filenames:

* **Class 0: Drowsy** (Filenames start with 'A', e.g., `A0001.png`)
* **Class 1: Non-Drowsy** (Filenames start with 'a', e.g., `a0001.png`)

### Key Challenge: Data Imbalance
A critical aspect of this dataset is its severe imbalance:
* **Drowsy (0):** 40,541 images
* **Non-Drowsy (1):** 1,252 images

To handle this, a **stratified split** was used to ensure that the training, validation, and test sets all contain the same proportional representation of each class.

* **Training Set:** 29,255 images
* **Validation Set:** 6,269 images
* **Test Set:** 6,269 images

---

## 🔧 Solution Methodology

The project is broken down into three main stages: Data Pipeline, Model Architecture, and Training.

### 1. Data Pipeline (`CustomDrowsyDataset` & `DataLoader`)

* **`CustomDrowsyDataset`:** A custom `Dataset` class was built to:
    1.  Load an image path.
    2.  Open the image using `PIL` and convert it to 'RGB'.
    3.  Assign a label (`0` or `1`) based on the first letter of the filename ('A' vs 'a').
    4.  Apply the defined data transformations.

* **Data Transformations:**
    * `Resize((224, 224))`: Resize all images to $224 \times 224$, the expected input size for ResNet.
    * `RandomHorizontalFlip(p=0.5)`: A simple data augmentation step to improve model robustness.
    * `ToTensor()`: Convert images to PyTorch tensors.
    * `Normalize(...)`: Normalize images using standard ImageNet mean and std dev.

* **`DataLoaders`:** Standard `DataLoader` objects were created for training, validation, and testing, with `shuffle=True` for the training loader.

### 2. Model Architecture (Transfer Learning)

1.  **Load Pretrained Model:** A `ResNet-18` model, pretrained on ImageNet, was loaded.
    ```python
    model = models.resnet18(pretrained=True)
    ```
2.  **Freeze Weights:** All parameters in the base model were frozen to prevent them from being updated during training. This preserves the powerful, general-purpose features learned from ImageNet.
    ```python
    for param in model.parameters():
        param.requires_grad = False
    ```
3.  **Replace Classifier Head:** The final fully-connected layer (`model.fc`) was replaced with a new `nn.Linear` layer.
    * **`in_features`:** Kept the original `num_ftrs` from ResNet-18 (which is 512).
    * **`out_features`:** Set to **1**. This single output neuron provides a "logit" for binary classification.

    ```python
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    ```

### 3. Training & Evaluation

* **Loss Function:** `nn.BCEWithLogitsLoss()`. This loss function is ideal for binary classification with a single logit output. It combines a Sigmoid layer and Binary Cross-Entropy in one stable class.
* **Optimizer:** `optim.Adam`. Crucially, the optimizer was *only* given the parameters of the new, unfrozen layer (`model.fc.parameters()`).
* **Training Loop:** The model was trained for **10 epochs**, validating at the end of each epoch.
* **Metrics:** Loss (Train/Val) and Accuracy (Val) were tracked.

---

## 🛠️ Tools & Libraries

* **Core:** `torch`, `torchvision`
* **Data Handling:** `numpy`, `sklearn` (for `train_test_split` and metrics)
* **Image Loading:** `PIL (Pillow)`
* **Visualization:** `matplotlib`, `seaborn` (for the confusion matrix)
* **Utilities:** `os`, `glob`, `tqdm` (for progress bars)

---

## 🎉 Results: 100% Accuracy

The model achieved **perfect scores** on both the validation and test datasets after just a few epochs.

This remarkable result suggests that the features extracted by the pretrained ResNet-18 are *exceptionally* effective for this dataset, allowing the new linear classifier to find a perfect separating boundary between the "Drowsy" and "Non-Drowsy" classes.

### Test Set Evaluation

The final evaluation on the held-out test set confirms the perfect performance.

**Classification Report:**

Here is a detailed, colorful README.md file based on the complete PyTorch script you provided.

Markdown

# 😴 Drowsiness Detection with PyTorch & ResNet-18

![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Language](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Model](https://img.shields.io/badge/Model-ResNet18-green.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen.svg)

This repository contains a PyTorch solution for a binary image classification task: **detecting drowsiness**.

The project uses **transfer learning** by leveraging a pretrained **ResNet-18** model, freezing its weights, and training only a new classifier head. The model is trained on a custom dataset where drowsiness is determined by the image's filename.

---

## 💾 The Dataset

The dataset (from `/kaggle/input/drowsy`) consists of thousands of images, categorized into two classes based on their filenames:

* **Class 0: Drowsy** (Filenames start with 'A', e.g., `A0001.png`)
* **Class 1: Non-Drowsy** (Filenames start with 'a', e.g., `a0001.png`)

### Key Challenge: Data Imbalance
A critical aspect of this dataset is its severe imbalance:
* **Drowsy (0):** 40,541 images
* **Non-Drowsy (1):** 1,252 images

To handle this, a **stratified split** was used to ensure that the training, validation, and test sets all contain the same proportional representation of each class.

* **Training Set:** 29,255 images
* **Validation Set:** 6,269 images
* **Test Set:** 6,269 images

---

## 🔧 Solution Methodology

The project is broken down into three main stages: Data Pipeline, Model Architecture, and Training.

### 1. Data Pipeline (`CustomDrowsyDataset` & `DataLoader`)

* **`CustomDrowsyDataset`:** A custom `Dataset` class was built to:
    1.  Load an image path.
    2.  Open the image using `PIL` and convert it to 'RGB'.
    3.  Assign a label (`0` or `1`) based on the first letter of the filename ('A' vs 'a').
    4.  Apply the defined data transformations.

* **Data Transformations:**
    * `Resize((224, 224))`: Resize all images to $224 \times 224$, the expected input size for ResNet.
    * `RandomHorizontalFlip(p=0.5)`: A simple data augmentation step to improve model robustness.
    * `ToTensor()`: Convert images to PyTorch tensors.
    * `Normalize(...)`: Normalize images using standard ImageNet mean and std dev.

* **`DataLoaders`:** Standard `DataLoader` objects were created for training, validation, and testing, with `shuffle=True` for the training loader.

### 2. Model Architecture (Transfer Learning)

1.  **Load Pretrained Model:** A `ResNet-18` model, pretrained on ImageNet, was loaded.
    ```python
    model = models.resnet18(pretrained=True)
    ```
2.  **Freeze Weights:** All parameters in the base model were frozen to prevent them from being updated during training. This preserves the powerful, general-purpose features learned from ImageNet.
    ```python
    for param in model.parameters():
        param.requires_grad = False
    ```
3.  **Replace Classifier Head:** The final fully-connected layer (`model.fc`) was replaced with a new `nn.Linear` layer.
    * **`in_features`:** Kept the original `num_ftrs` from ResNet-18 (which is 512).
    * **`out_features`:** Set to **1**. This single output neuron provides a "logit" for binary classification.

    ```python
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    ```

### 3. Training & Evaluation

* **Loss Function:** `nn.BCEWithLogitsLoss()`. This loss function is ideal for binary classification with a single logit output. It combines a Sigmoid layer and Binary Cross-Entropy in one stable class.
* **Optimizer:** `optim.Adam`. Crucially, the optimizer was *only* given the parameters of the new, unfrozen layer (`model.fc.parameters()`).
* **Training Loop:** The model was trained for **10 epochs**, validating at the end of each epoch.
* **Metrics:** Loss (Train/Val) and Accuracy (Val) were tracked.

---

## 🛠️ Tools & Libraries

* **Core:** `torch`, `torchvision`
* **Data Handling:** `numpy`, `sklearn` (for `train_test_split` and metrics)
* **Image Loading:** `PIL (Pillow)`
* **Visualization:** `matplotlib`, `seaborn` (for the confusion matrix)
* **Utilities:** `os`, `glob`, `tqdm` (for progress bars)

---

## 🎉 Results: 100% Accuracy

The model achieved **perfect scores** on both the validation and test datasets after just a few epochs.

This remarkable result suggests that the features extracted by the pretrained ResNet-18 are *exceptionally* effective for this dataset, allowing the new linear classifier to find a perfect separating boundary between the "Drowsy" and "Non-Drowsy" classes.

### Test Set Evaluation

The final evaluation on the held-out test set confirms the perfect performance.

**Classification Report:**
--- Classification Report --- precision recall f1-score support

  Drowsy       1.00      1.00      1.00      6081
Non Drowsy 1.00 1.00 1.00 188

accuracy                           1.00      6269
macro avg 1.00 1.00 1.00 6269 weighted avg 1.00 1.00 1.00 6269


**Confusion Matrix:**
The confusion matrix shows **zero false positives and zero false negatives** on the 6,269 test images.
