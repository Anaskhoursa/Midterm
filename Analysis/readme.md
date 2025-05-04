# CIC-IDS2017 Data Mining Project

This project applies machine learning techniques to detect cyber threats using the CIC-IDS2017 dataset. It includes data preprocessing, exploratory data analysis (EDA), and the implementation of a Random Forest classifier to classify network traffic as benign or malicious.

## Overview

The objective of this project is to demonstrate a practical application of data mining techniques in the field of cybersecurity. By using the CIC-IDS2017 dataset, we aim to build a classification model capable of identifying network intrusions. The project follows a structured approach involving data preprocessing, EDA, model training, and performance evaluation.

## Files Included

 - `analysis.py`: Python script that performs data loading, preprocessing, analysis, model training, and evaluation.
 - `midterm report.pdf`: Report summarizing the methodology and results.
 - `readme.md`: This documentation file.

## Requirements

Before running the code, make sure you have the following Python libraries installed:


pip install pandas numpy matplotlib seaborn scikit-learn

## How to run

-Place dataset csv file in the same directory as analysis.py (download it from here: [Download CIC-IDS2017 Dataset](https://drive.google.com/file/d/1_1yKjeXzgjLef4LDhag3xMPaQklEomUP/view?usp=drive_link))

-Open a terminal in that directory.

-Run the script using:
python analysis.py


# This will:

Load and preprocess the dataset

Train a Random Forest model

Evaluate its performance

Display visualizations and metrics



## Code Explanation
# Data Preprocessing
Unnecessary and constant columns are removed.

Rows with infinite or missing values are filtered out.

Labels are encoded (Benign = 0, DDoS = 1).

Features are scaled using StandardScaler to normalize values.

The final dataset is split into training (80%) and testing (20%) sets.

# Exploratory Data Analysis (EDA)
Displays label distribution using a countplot.

Shows correlation between features using a heatmap.

Provides descriptive statistics of the dataset.


# Model Training & Evaluation
A Random Forest Classifier is trained.

Predictions are made on the test set.

Evaluation includes:

-Accuracy score

-Classification report (precision, recall, F1-score)

-Confusion matrix (plotted with Seaborn)

## Usage
This project is useful for:

-Practicing supervised learning on cybersecurity datasets

-Understanding feature preprocessing and model evaluation

-Developing foundational knowledge for intrusion detection systems (IDS)

