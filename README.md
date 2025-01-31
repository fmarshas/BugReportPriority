# BugReportPriority
Predicting software bug report priority (P1–P5) through machine learning (ML) or deep learning (DL) techniques (Kaggle ISEC Data Challenge)
# Bug-Priority Prediction

This repository contains code and data for predicting software bug report priority (P0 to P4). We take inspiration from the Kaggle challenge at [https://www.kaggle.com/competitions/isec-sdc-2025/data](https://www.kaggle.com/competitions/isec-sdc-2025/data) which defines the problem setup and provides sample datasets.

## Dataset Overview

- **train.csv**  
  This file holds ~53,000 rows of labeled bug reports. Each row includes:
  - **Issue_id:** A unique identifier for the bug report.
  - **Title:** A brief title summarizing the bug (4–5 words typically).
  - **Description:** A more detailed explanation, often including error messages or steps to reproduce.
  - **Component, Status, Resolution:** Categorical fields describing where the issue belongs, its progress state, and final outcome.
  - **Priority:** The true bug priority label (`0` = highest urgency, `4` = lowest).  

In our project, we use **train.csv** as follows:
1. **Split** into training, validation, and test partitions (70% / 10% / 20%).
2. **Remove** any null records.
3. **Balance** classes by oversampling minority priorities, ensuring that each priority label is represented more evenly.

We do **not** use `test.csv` from Kaggle because the official “hidden” answers are unavailable, complicating local evaluation.

## Code

- **pnyb** (Notebook / Python code)  
  The core pipeline for data preprocessing, feature engineering, and model training. Main steps include:
  1. **Text Merging & Encoding:** We concatenate Title and Description into a single text column, then build:
     - BERT embeddings (via a pretrained `BertModel`).
     - TF-IDF vectors (`TfidfVectorizer`).
  2. **Categorical One-Hot Encoding:** For Component, Status, and Resolution fields.
  3. **Oversampling:** To address class imbalance in Priority.
  4. **Gradient Boosting** (XGBoost, LightGBM) for classification metrics (F1, confusion matrices).
  5. **LambdaMART** for ranking metrics (NDCG, MRR).

Inside the code notebook/file, you’ll find detailed commentary on each step, including how we load the data, generate embeddings, train the models, and evaluate performance.

## Getting Started

1. Clone the repository and place `train.csv` in the root directory (or update file paths in the notebook).
2. Install the required dependencies (e.g., `transformers`, `lightgbm`, `xgboost`, `scikit-learn`, etc.).
3. Run the notebook / script to reproduce the data splits, training, and evaluation steps.

## Acknowledgments

- **Kaggle ISEC-SDC-2025**: We draw on the challenge specifications and data from the [Kaggle competition](https://www.kaggle.com/competitions/isec-sdc-2025).
- **HuggingFace Transformers**: Provides pretrained BERT models for text embeddings.
- **XGBoost, LightGBM**: Gradient boosting frameworks used for classification and ranking tasks.

Please contact us or open an issue if you have any questions or suggestions.

