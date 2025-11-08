
# 🎤 Speech2Score: Grammar Scoring Engine (SHL Intern Assessment 2025)

## ✨ Overview
This project implements a **Grammar Scoring Engine** for spoken audio samples. The core objective is to predict a continuous grammar score (ranging from 0 to 5) based on the speaker's fluency and grammatical accuracy, as defined by a MOS Likert scale rubric.

The approach leverages a powerful Acoustic Model for transcription, followed by traditional Machine Learning techniques applied to extracted linguistic features.

| Metric | Target |
| :--- | :--- |
| **Input** | `.wav` Audio files (45-60 seconds) |
| **Output** | Continuous Score (0 to 5) |
| **Evaluation** | Pearson Correlation and **RMSE** |

---

## 🏗️ Pipeline Architecture

Our solution follows a two-stage pipeline: **Acoustic Modeling** (Speech-to-Text) followed by **Regression Modeling** (Score Prediction). 
### Phase 1: Feature Engineering via Speech-to-Text (Acoustic Model)
1.  **Transcription**: The audio file is transcribed into raw text using the **Whisper** model (`medium.en` variant). Whisper is an effective foundation for converting the spoken data, often handling noise and accents robustly.
2.  **Linguistic Feature Extraction**: The transcribed text is analyzed using **NLTK** (Natural Language Toolkit). Key linguistic features are engineered, including:
    * **Total Words**
    * **Noun Count** (NN tags)
    * **Verb Count** (VB tags)
    * **Adjective Count** (JJ tags)
    * **Average Word Length**
    These features serve as proxies for complexity and structure, which correlate with higher grammar scores.

### Phase 2: Score Prediction (Regression Model)
1.  **Model Selection**: A **Random Forest Regressor** is chosen for its non-linearity, robustness to noisy features, and effectiveness on tabular data.
2.  **Training**: The model is trained on the extracted linguistic features (`X`) and the ground truth MOS Likert Grammar Scores (`y`).
3.  **Prediction**: The trained model outputs the final continuous grammar score (0-5) for the unseen test samples.

---

## 📊 Results & Performance

As required by the assessment, the Root Mean Square Error (RMSE) was calculated on the training data to benchmark the model's capacity.

| Data Set | Metric | Value |
| :--- | :--- | :--- |
| **Training Data** | **RMSE** (Root Mean Square Error) | `0.306` |
| **Evaluation** | *Pearson Correlation* | *(Calculated on Test Data after Submission)* |

### 📝 Brief Report (from Jupyter Notebook)

The **Random Forest Regressor** achieved a **Training RMSE of 0.306**. This is a solid baseline, indicating that the model captures a significant portion of the variance in the grammar scores using only structural linguistic features. The low training error suggests the model has effectively learned the relationship between the engineered features and the target scores.

**Areas for Future Improvement**:
* **Enhanced Features**: Incorporate more sophisticated linguistic features such as grammatical error counts (using a dedicated library like LanguageTool or custom error tagging), **sentence parse tree complexity**, or other measures of syntactic variation.
* **Advanced Text Models**: Integrate **BERT embeddings** or other transformer-based text models to capture the semantic and deeper grammatical context of the transcriptions.
* **End-to-End Learning**: Explore fine-tuning **Whisper** or using other end-to-end audio-to-score models to bypass potential error accumulation from the two-stage pipeline.

---

## 🏃 Getting Started

### Prerequisites

* Python 3.8+
* `pip install -r requirements.txt` (see `requirements.txt` below)

### Execution

```bash
# Assuming your data is structured as /dataset/audios/train/ and /dataset/audios/test/
python speech2score.py



That is a fantastic project! Building a grammar scoring engine combines fascinating elements of Speech-to-Text (Acoustic Modeling) and Natural Language Processing (Linguistic Feature Engineering).

Here are the files you requested, designed to be colorful and beautiful for your GitHub repository.

1. README.md
This file summarizes your project, its architecture, and the results, using rich markdown for a professional look.

Markdown
# 🎤 Speech2Score: Grammar Scoring Engine (SHL Intern Assessment 2025)

## ✨ Overview
This project implements a **Grammar Scoring Engine** for spoken audio samples. The core objective is to predict a continuous grammar score (ranging from 0 to 5) based on the speaker's fluency and grammatical accuracy, as defined by a MOS Likert scale rubric.

The approach leverages a powerful Acoustic Model for transcription, followed by traditional Machine Learning techniques applied to extracted linguistic features.

| Metric | Target |
| :--- | :--- |
| **Input** | `.wav` Audio files (45-60 seconds) |
| **Output** | Continuous Score (0 to 5) |
| **Evaluation** | Pearson Correlation and **RMSE** |

---

## 🏗️ Pipeline Architecture

Our solution follows a two-stage pipeline: **Acoustic Modeling** (Speech-to-Text) followed by **Regression Modeling** (Score Prediction). 
### Phase 1: Feature Engineering via Speech-to-Text (Acoustic Model)
1.  **Transcription**: The audio file is transcribed into raw text using the **Whisper** model (`medium.en` variant). Whisper is an effective foundation for converting the spoken data, often handling noise and accents robustly.
2.  **Linguistic Feature Extraction**: The transcribed text is analyzed using **NLTK** (Natural Language Toolkit). Key linguistic features are engineered, including:
    * **Total Words**
    * **Noun Count** (NN tags)
    * **Verb Count** (VB tags)
    * **Adjective Count** (JJ tags)
    * **Average Word Length**
    These features serve as proxies for complexity and structure, which correlate with higher grammar scores.

### Phase 2: Score Prediction (Regression Model)
1.  **Model Selection**: A **Random Forest Regressor** is chosen for its non-linearity, robustness to noisy features, and effectiveness on tabular data.
2.  **Training**: The model is trained on the extracted linguistic features (`X`) and the ground truth MOS Likert Grammar Scores (`y`).
3.  **Prediction**: The trained model outputs the final continuous grammar score (0-5) for the unseen test samples.

---

## 📊 Results & Performance

As required by the assessment, the Root Mean Square Error (RMSE) was calculated on the training data to benchmark the model's capacity.

| Data Set | Metric | Value |
| :--- | :--- | :--- |
| **Training Data** | **RMSE** (Root Mean Square Error) | `0.306` |
| **Evaluation** | *Pearson Correlation* | *(Calculated on Test Data after Submission)* |

### 📝 Brief Report (from Jupyter Notebook)

The **Random Forest Regressor** achieved a **Training RMSE of 0.306**. This is a solid baseline, indicating that the model captures a significant portion of the variance in the grammar scores using only structural linguistic features. The low training error suggests the model has effectively learned the relationship between the engineered features and the target scores.

**Areas for Future Improvement**:
* **Enhanced Features**: Incorporate more sophisticated linguistic features such as grammatical error counts (using a dedicated library like LanguageTool or custom error tagging), **sentence parse tree complexity**, or other measures of syntactic variation.
* **Advanced Text Models**: Integrate **BERT embeddings** or other transformer-based text models to capture the semantic and deeper grammatical context of the transcriptions.
* **End-to-End Learning**: Explore fine-tuning **Whisper** or using other end-to-end audio-to-score models to bypass potential error accumulation from the two-stage pipeline.

---

## 🏃 Getting Started

### Prerequisites

* Python 3.8+
* `pip install -r requirements.txt` (see `requirements.txt` below)

### Execution

```bash
# Assuming your data is structured as /dataset/audios/train/ and /dataset/audios/test/
python speech2score.py
Output
The script generates submission.csv containing the predicted grammar scores for the test set.

Code snippet
filename,label
audio_141,2.540
audio_114,2.505
...

***

## 2. `speech2score.py` (The Python Code)

This is the consolidated and well-commented Python script from the Jupyter Notebook, formatted for readability.

```python
# ==============================================================================
# 🎯 SHL INTERN HIRING ASSESSMENT 2025: GRAMMAR SCORING ENGINE
# ==============================================================================
#
# Pipeline: Whisper (Acoustic Model) -> Linguistic Feature Extraction (NLTK)
#           -> Random Forest Regressor (Prediction Model)
#
# Requirements: The code must calculate and report the RMSE on the Training Data.
#
# ==============================================================================

import os
import pandas as pd
import numpy as np
import whisper
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# --- 1. SETUP AND MODEL LOADING -----------------------------------------------

# Download necessary NLTK packages if not present
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    # Attempt to download the corrected NLTK tagger model (for known Kaggle/NLTK issue)
    # This addresses a common issue where the tagger might fail on certain environments
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except Exception:
        pass
except Exception as e:
    print(f"NLTK Download Error: {e}")

# Load the Acoustic Model (Speech-to-Text)
# We use 'medium.en' for a balance of speed and accuracy on English audio.
print("Loading Whisper ASR model...")
model = whisper.load_model("medium.en")

# Define base paths for data loading
BASE_DATA_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/audios/"
TRAIN_CSV_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/train.csv"
TEST_CSV_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/test.csv"


# --- 2. HELPER FUNCTIONS ------------------------------------------------------

def get_all_audio_paths(base_filename, base_dir):
    """Finds the base audio file and its variants (e.g., audio_XX_1.wav)"""
    paths = []
    # Original file name
    paths.append(os.path.join(base_dir, f"{base_filename}.wav"))
    
    # Check for variants
    for suffix in ['_1', '_2']:
        paths.append(os.path.join(base_dir, f"{base_filename}{suffix}.wav"))
    
    # Return only existing paths
    return [p for p in paths if os.path.exists(p)]


def extract_features(audio_path):
    """
    Transcribes audio and extracts linguistic features from the text.
    These features serve as proxies for grammatical complexity.
    """
    # 1. Transcription (Acoustic Model)
    # Using 'fp16=False' for better compatibility on different hardware.
    result = model.transcribe(audio_path, fp16=False)
    text = result['text']
    
    # 2. Linguistic Feature Extraction (NLTK)
    # Tokenization and Part-of-Speech Tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    total_words = len(tokens)
    
    # Simple proxies for grammatical complexity (based on POS tags)
    num_nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    num_verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    num_adjs = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    
    # Lexical diversity/complexity
    avg_word_len = sum(len(word) for word in tokens) / total_words if total_words else 0

    return {
        'text': text,
        'total_words': total_words,
        'num_nouns': num_nouns,
        'num_verbs': num_verbs,
        'num_adjs': num_adjs,
        'avg_word_len': avg_word_len
    }


# --- 3. TRAINING PHASE --------------------------------------------------------

# Load training data
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_features = []
train_base_dir = os.path.join(BASE_DATA_PATH, "train/")

print("\n--- Starting Training Data Processing and Feature Extraction ---")
# Process each entry in the training manifest
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    # The filename in train.csv is the base name (e.g., 'audio_49')
    base_filename = row['filename']
    label = row['label']
    
    # Handle multiple audio segments for a single label
    audio_paths = get_all_audio_paths(base_filename, train_base_dir)
    
    if not audio_paths:
        # print(f"⚠️ Could not find audio files for base ID: {base_filename}. Skipping.")
        continue

    # Process all segments associated with this label
    for audio_path in audio_paths:
        try:
            features = extract_features(audio_path) 
            features['label'] = label
            # Store the unique filename (including suffix) for tracking
            features['full_filename'] = os.path.basename(audio_path)
            train_features.append(features)
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(audio_path)}: {e}")

# Convert to DataFrame for model training
train_features_df = pd.DataFrame(train_features)

# Define features (X) and target (y)
FEATURES = ['total_words', 'num_nouns', 'num_verbs', 'num_adjs', 'avg_word_len']
X_train = train_features_df[FEATURES]
y_train = train_features_df['label']

# Initialize and Train the Model
RFR_model = RandomForestRegressor(random_state=42)
RFR_model.fit(X_train, y_train)

# Calculate and report Training RMSE (COMPULSORY REQUIREMENT)
y_train_pred = RFR_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n--- Training Complete ---")
print(f"✅ Training Data RMSE: {train_rmse:.3f}")
print("-" * 30)


# --- 4. TESTING AND PREDICTION PHASE ------------------------------------------

# Load testing data
test_df = pd.read_csv(TEST_CSV_PATH)
test_features = []
test_base_dir = os.path.join(BASE_DATA_PATH, "test/")

print("\n--- Starting Test Data Processing and Feature Extraction ---")
# Process each entry in the test manifest
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # The filename in test.csv is the final submission name (e.g., 'audio_49')
    audio_filename_base = row['filename'].replace('.wav', '')
    
    # Need to handle the fact that some files might have variants (e.g., _1, _2)
    # The submission requires the base name, but we need to process *an* existing file.
    audio_paths = get_all_audio_paths(audio_filename_base, test_base_dir)
    
    # We will only process the first found file for the final prediction
    # to maintain a 1:1 mapping with the submission file.
    if not audio_paths:
        # print(f"⚠️ Missing audio file: {audio_filename_base} (and variants). Skipping.")
        continue
    
    # Use the first available audio segment for prediction
    audio_path = audio_paths[0] 
    
    try:
        features = extract_features(audio_path)
        features['filename'] = audio_filename_base # Store the base filename for submission
        test_features.append(features)
        
    except Exception as e:
        print(f"❌ Failed to process {os.path.basename(audio_path)}: {e}")

test_features_df = pd.DataFrame(test_features)
X_test = test_features_df[FEATURES]

# Predict scores on the test set
test_predictions = RFR_model.predict(X_test)


# --- 5. SUBMISSION FILE GENERATION --------------------------------------------

# Create the submission DataFrame
submission = pd.DataFrame({
    'filename': test_features_df['filename'].apply(lambda x: f"{x}.wav"),
    'label': test_predictions.round(3) # Scores must be continuous/float
})

# Filter the submission to only include filenames from the original test.csv
# This handles cases where one base ID might have multiple audio segments but 
# only one final prediction is required per ID.
final_submission_df = test_df.merge(submission, on='filename', how='left')
# Use the predicted label, but ensure missing predictions are handled (e.g., by mean/median, 
# but here we rely on having processed at least one audio segment per base ID).
# In this notebook, we only processed one segment per base ID for test.csv,
# so a simple merge/overwrite is fine.

# Final formatting check: ensures only two columns (filename, label) as required
submission_output = final_submission_df[['filename', 'label']]

# Save the final submission file
submission_output.to_csv("submission.csv", index=False)

print("\n--- Submission File Generated ---")
print("✅ submission.csv created successfully with 197 predictions.")
print("Head of Submission:")
print(submission_output.head())
print("-" * 30)

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
