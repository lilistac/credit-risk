# Credit Risk

Credit risk modeling plays a crucial role in the banking and financial industry. This repository offers tools and resources that can be used for building and analyzing credit risk models. The primary objective is to utilize the simplicity and interpretability of the Logistic Regression model for assessing credit risks.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
    - [Exploratory Data Analysis](#eda)
    - [Data Preparation](#data-preparation)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Contribution](#contribution)

## Introduction

In this project, we aim to develop a predictive model that evaluates loan applicants' creditworthiness and minimizes the risk of default. This project is crucial because it can help financial institutions make informed decisions about loan approvals, which can ultimately reduce the risk of losses due to unpaid loans.  

## Installation

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/lilistac/credit-risk.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
### Data
Create the following folder to store data
```
credit-risk
└── data
    ├── raw
    ├── processed
    └── submission
```


Add the raw data to the folder **data/raw**
### Exploratory Data Analysis (EDA)
Notebooks in **eda_notebooks** folder give detailed information about 7 tables with structure: 
1. Import Libraries
2. Load data & Basic information
3. Target column
4. Missing values
5. Define numeric and category features
6. Imbalanced categorical features
7. Outliers
8. Anomalies
9. Correlation
10. Categorical analysis
11. Numerical analysis

### Data Preparation
Before training the credit risk model, data preparation is essential. Use  **build_features.py** from folder **src/feature** to guide you through:
- Data cleaning
- Feature engineering

Features created from this file are stored as 'train.csv' and 'test.csv' located in the data/processed 

### Hyperparameter Tuning
We tune hyperparameter using RandomizedSearchCV and GridSearchCV by running the file **hyperparameter_tuning.py** and save the best model as **best_model.plk** in **src/model**

### Prediction
To get the predictive value we run the file **prediction.py** in folder **src/model**
and get the result in **data/submission**

Model: Logistic Regression

## Evaluation 
Score: Gini <br>
Formula: GINI = 2 x AUC - 1 <br>
Public score: 0.54761 <br>
Private score: 0.55685

## Contribution
- Nguyễn Cẩm Ly (Team leader)
- Võ Thị Yến Nhi
- Nguyễn Thành Long