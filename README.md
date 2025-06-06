# RLDatix - Predictive Modelling and Named Entity Recognition

This project applies machine learning and LLM to healthcare data to achieve two objectives:

1. **Predicting 30-day patient readmissions** using structured clinical data.
2. **Extracting clinical entities** from unstructured discharge notes using transformer-based Named Entity Recognition (NER).

---
## Overview of Each Application

### 1\. Patient Readmission Prediction
- Preprocessing of structured features (categorical and numerical)
- Class imbalance handled using SMOTE
- Feature correlation analysis and selection
- Model: Random Forest classifier
- Evaluation on training & test sets for comparison:
    - classification report (accuracy, recall, precision, F1 score for both classes)
    - ROC AUC

### 2\. Named Entity Recognition with LLM
- Load discharge notes from the dataset
- Design and format prompts for medical entity extraction
- Use transformers library to run inference with a pre-trained T5 model
- Output structured entity lists

## Project Structure

```
.
├── RLDatix-Patient_Readmission_Prediction.ipynb  # Patient Readmission Prediction notebook
├── RLDatix-Named_Entity_Recognition_LLM.ipynb    # NER notebook
├── README.md                                     # This file
├── requirements.txt                              # Contains a list of packages or libraries needed   
└── data/                                         # Directory for raw/processed data files
    └── <Assignment_Data.csv>                     # Contains the project dataset
```

---

## Requirements
```bash
conda create -n <virtual environment name> python=3.9 # create a virtual environment
conda activate <virtual environment name> # activate the virtual environment
```
then 

```bash
pip install -r requirements.txt
```
or

```bash
pip install pandas numpy matplotlib seaborn statsmodels
pip install -U scikit-learn imbalanced-learn
pip install transformers torch

