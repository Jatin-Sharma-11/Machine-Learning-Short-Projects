# CSIRO - Image2Biomass Prediction Solution

This repository contains my solution for the [CSIRO - Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass) competition on Kaggle.

**Goal:** Build a model to predict pasture biomass from images. The model predicts 5 separate target values, which are evaluated using a weighted $R^2$ score.

## 🚀 My Approach

This solution uses a PyTorch-based Deep Learning pipeline to perform multi-target regression directly from the pasture images. The entire pipeline is contained within the main notebook.

### 1. Data Preprocessing

* **Wide Format:** The `train.csv` is pivoted from a long format (1 row per image-target pair) to a "wide" format (1 row per unique image) containing 5 target columns.
* **Target Transform:** The 5 target variables (`Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `GDM_g`, `Dry_Total_g`) are all heavily right-skewed. A **`log1p` transform (`log(x + 1)`)** is applied to each target to make their distributions more normal, which significantly helps the model learn.
* **Data Split:** The 357 training images are split into an 80% training set (285 images) and a 20% validation set (72 images).

### 2. Augmentation & DataLoaders

Due to the very small dataset, heavy data augmentation is critical.

* **Transforms:**
    * `Resize` (to 224x224, the expected input for EfficientNet)
    * `RandomHorizontalFlip`
    * `RandomVerticalFlip`
    * `RandomRotation(15)`
    * `ColorJitter` (brightness, contrast, saturation, hue)
    * `ToTensor`
    * `Normalize` (with ImageNet statistics)
* **Dataset:** A custom PyTorch `Dataset` class is used to load images, apply the correct transforms (train vs. val), and return the image tensor and its 5-target label vector.

### 3. Model Architecture

* **Transfer Learning:** A pre-trained **EfficientNet-B0** model (from `torchvision.models`) is used as the feature-extraction backbone.
* **Regression Head:** The final classification layer (`classifier`) is replaced with a custom regression head to output 5 values:
    ```python
    nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5)
    )
    ```

### 4. Training

* **Loss Function:** A **Weighted Mean Squared Error (MSE) Loss** is used. This loss function directly mirrors the competition's evaluation metric by applying the specified weights to the MSE of each target:
    * `Dry_Total_g`: 0.5
    * `GDM_g`: 0.2
    * Others: 0.1
* **Optimizer:** `AdamW` (learning rate: `1e-4`).
* **Strategy:** The model is trained for **20 Epochs**, saving only the model weights that achieve the lowest validation loss.

### 5. Prediction

1.  The best saved model (`best_model.pth`) is loaded.
2.  Predictions are made on the test images (which also undergo `Resize`, `ToTensor`, and `Normalize`).
3.  The model outputs 5 log-transformed values.
4.  An **`expm1` (`exp(x) - 1`)** function is applied to the predictions to reverse the `log1p` transform, converting them back to the original biomass (grams) scale.
5.  The "wide" predictions are melted back into a "long" format and merged with the `test.csv` to create the final `submission.csv` file.

## ⚙️ How to Run

1.  **Data:** Download the competition data from [Kaggle](https://www.kaggle.com/competitions/csiro-biomass) and place it in a directory.
2.  **Configuration:** In **Cell 2**, update the `BASE_PATH` variable to point to your data directory (e.g., `/kaggle/input/csiro-biomass/`).
3.  **Notebook:** Open the main notebook in a GPU-enabled environment (e.g., Kaggle, Colab).
4.  **Execute:** Run all cells sequentially. The `best_model.pth` file will be saved, and a `submission.csv` file will be generated in the root directory.

## 📊 Results

* **Final Validation Loss (Weighted MSE):** **0.4802**
* **Public LB Score:** [TODO: Fill in your LB score after submission]
* **Private LB Score:** [TODO: Fill in your final score]

---

<details>
<summary>Click to view Full Competition Details</summary>

### Description
Farmers often walk into a paddock and ask one question: “Is there enough grass here for the herd?” Pasture biomass - the amount of feed available - shapes when animals can graze, when fields need a break, and how to keep pastures productive season after season. This competition challenges you to build a model that predicts pasture biomass from images, ground-truth measures, and publicly available datasets.

### Evaluation
The model performance is evaluated using a weighted average of $R^2$ scores across the five output dimensions. The final score is calculated as:

$$
\text{Final Score} = \sum_{i=1}^{5} w_i \cdot R_i^2
$$

Where:
* $R_i^2$ represents the coefficient of determination for dimension $i$
* The weights $w_i$ used are:
    * `Dry_Green_g`: 0.1
    * `Dry_Dead_g`: 0.1
    * `Dry_Clover_g`: 0.1
    * `GDM_g`: 0.2
    * `Dry_Total_g`: 0.5

### Submission File
Submit a CSV in long format with exactly two columns: `sample_id` and `target`.

| sample_id | target |
|---|---|
| ID1001187975__Dry_Green_g | 0.0 |
| ID1001187975__Dry_Dead_g | 0.0 |
| ID1001187975__Dry_Clover_g | 0.0 |
| ID1001187975__GDM_g | 0.0 |
| ID1001187975__Dry_Total_g | 0.0 |

### Code Requirements
* CPU Notebook <= 9 hours run-time
* GPU Notebook <= 9 hours run-time
* Internet access disabled
* Freely & publicly available external data is allowed, including pre-trained models
* Submission file must be named `submission.csv`

</details>
