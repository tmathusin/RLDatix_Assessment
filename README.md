# RLDatix - Predictive Modelling and Named Entity Recognition

This project applies machine learning and LLM to healthcare data to achieve two objectives:

1. **Predicting 30-day patient readmissions** using structured clinical data.
2. **Extracting clinical entities** from unstructured discharge notes using transformer-based Named Entity Recognition (NER).

---

## 📂 Project Structure

- `RLDatix-Patient_Readmission_Prediction.ipynb`  
  Builds and evaluates a machine learning model to predict the likelihood of patient readmission within 30 days.

- `RLDatix-Named_Entity_Recognition_LLM.ipynb`  
  Uses prompt-based large language models (LLMs) for extracting named entities from discharge summaries.

- `Assignment_Data.csv`  
  Contains patient data with both structured features and free-text notes used across both applications.

---
## Requirements: 
```bash
pip install pandas numpy matplotlib seaborn statsmodels
pip install -U scikit-learn imbalanced-learn
pip install transformers torch

---
## 🔍 Overview of Each Application

### 1. Patient Readmission Prediction
- Preprocessing of structured features (categorical and numerical)
- Class imbalance handled using SMOTE
- Feature correlation analysis and selection
- Model: Random Forest classifier
- Evaluation on training & test sets for comparison:
    - classification report (accuracy, recall, precision, F1 score for both classes)
    - ROC AUC

### 2. Named Entity Recognition with LLM
- Load discharge notes from the dataset
- Design and format prompts for medical entity extraction
- Use transformers library to run inference with a pre-trained T5 model
- Output structured entity lists
