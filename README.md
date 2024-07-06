# 🤖 First ML Project 💻

This repository contains my first machine learning project, which predicts the solubility of compounds using linear regression and random forest regression models. The project includes data preparation, model building, evaluation, and visualization.

## Project Overview

### 📄 Load Data
The dataset used is the Delaney solubility dataset, which contains chemical compound descriptors and their corresponding solubility values (logS). The dataset is fetched directly from a URL using pandas.

### 🛠 Data Preparation
The dataset is split into features (`x`) and the target variable (`y`). The features include various chemical descriptors such as molecular weight, number of rings, and number of bonds. The data is further split into training and testing sets using a 80-20 split.

### 🤖 Model Building
#### Linear Regression
- **Training the Model**: A linear regression model is trained on the training data to predict the solubility (`logS`).
- **Model Evaluation**: The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) on both training and testing sets.

#### Random Forest Regression
- **Training the Model**: A random forest regression model with a maximum depth of 2 is trained on the training data.
- **Model Evaluation**: Similar to linear regression, the random forest model's performance is evaluated using MSE and R2 scores on both training and testing sets.

### 📊 Model Comparison
The performance metrics (MSE and R2) of both models (linear regression and random forest) are compared to determine which model performs better for predicting solubility.

### 📈 Visualization
A scatter plot is generated to visualize the predicted vs. actual solubility values for the linear regression model. This plot helps in understanding how well the model predictions align with the actual experimental values.

## Files

- `main.py`: The main script containing all the code for data loading, preparation, model training, evaluation, and visualization.

## Getting Started

### Prerequisites

Make sure you have Python and the necessary libraries installed. You can install the required libraries using pip:

```sh
pip install pandas scikit-learn matplotlib numpy
