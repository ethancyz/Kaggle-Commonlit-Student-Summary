# Kaggle CommonLit – Evaluate Student Summaries

This repository contains my solution and code organization for the Kaggle competition  
**“CommonLit – Evaluate Student Summaries”**.

The goal of the competition is to automatically evaluate the quality of student-written summaries using natural language processing models.

---

## 📌 Competition Overview

In this competition, participants build models to score student summaries written for reading passages provided by **CommonLit**, a nonprofit educational organization.

Each summary is evaluated on two dimensions:

- **Content**: How well the summary captures the main ideas and key details of the source text.
- **Wording**: The clarity, precision, and fluency of the student's writing.

The task is a **regression problem**, and the official evaluation metric is **MCRMSE (Mean Columnwise Root Mean Squared Error)** over the two targets.

Competition link:  
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

---

## 📂 Dataset Description

The dataset consists of the following main files:

| File | Description |
|-----|-------------|
| `summaries_train.csv` | Training summaries with `content` and `wording` scores |
| `summaries_test.csv` | Test summaries without labels |
| `prompts_train.csv` | Source texts and questions for training |
| `prompts_test.csv` | Source texts and questions for testing |
| `sample_submission.csv` | Submission format example |

Each summary is linked to a prompt via `prompt_id`.

---

## 🧠 Overall Approach

The solution follows a **pretrained Transformer fine-tuning pipeline** with cross-validation.

### Main steps:

1. **Text preprocessing**
   - Clean raw text (remove URLs, HTML tags, special characters)
   - Tokenize summaries and prompts
   - Compute basic statistical features (length, n-gram overlap, spelling errors)

2. **Input construction**
   - Concatenate student summary with prompt information using special tokens:
     ```
     [Summary] [SEP] [Prompt Question Keywords] [SEP] [Prompt Title + Prompt Text Keywords]
     ```

3. **Model**
   - Pretrained model: `microsoft/deberta-v3-large`
   - Converted from classification to **regression**
   - Separate models trained for `content` and `wording`

4. **Training strategy**
   - **GroupKFold cross-validation** using `prompt_id` to prevent data leakage
   - Partial layer freezing to stabilize training and reduce overfitting
   - Early stopping based on validation RMSE

5. **Prediction**
   - Out-of-fold (OOF) predictions for validation
   - Test predictions averaged across folds

---

## ⚙️ Model Details

- Backbone: **DeBERTa v3 Large**
- Framework: HuggingFace Transformers
- Task type: Regression (`num_labels = 1`)
- Loss: Mean Squared Error
- Evaluation: RMSE / MCRMSE

### Layer Freezing

To improve stability and efficiency:

- Embedding layer is frozen
- First **N encoder layers** are frozen (default: `N = 18`)
- Only upper layers and regression head are fine-tuned

---

## 🧪 Evaluation Metric

The competition uses **MCRMSE**, defined as:

MCRMSE = mean(RMSE_content, RMSE_wording)

yaml
Copy code

This metric is implemented during validation to monitor performance.

---

## 📁 Repository Structure

.
├── train_single_label_weight_large_v1.ipynb # Main training notebook
├── data/ # Competition data
├── models/ # Saved models (by fold & target)
├── README.md

markdown
Copy code

Key components inside the notebook:

- `Preprocessor` – text cleaning and feature engineering
- `ContentScoreRegressor` – model wrapper for training and inference
- `train_by_fold()` – cross-validation training
- `validate()` – OOF prediction
- `predict()` – test-time inference

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ethancyz/Kaggle-Commonlit-Student-Summary
cd Kaggle-Commonlit-Student-Summary
2. Install dependencies
bash
Copy code
pip install transformers datasets nltk pyspellchecker scikit-learn accelerate
3. Train the model
Run the notebook step by step:

text
Copy code
train_single_label_weight_large_v1.ipynb
The notebook will:

Train separate models for content and wording

Perform 4-fold GroupKFold cross-validation

Save best models per fold

4. Generate predictions
The final output will be a submission.csv file in the required Kaggle format:

css
Copy code
student_id,content,wording
📊 Notes & Observations
Freezing lower layers significantly reduces GPU memory usage

Long input sequences (max_length ≈ 896) improve context coverage but increase training cost

Group-based CV is critical to avoid prompt-level leakage

Separate modeling of content and wording improves stability

📎 References
Kaggle Competition Page
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

HuggingFace Transformers
https://huggingface.co/docs/transformers

📜 License
This project is intended for educational and research purposes.
