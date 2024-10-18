# E-Commerce GMV Prediction Model

This project aims to predict **Gross Merchandise Value (GMV)** for an e-commerce company using machine learning techniques. The dataset contains various order-level data, and the goal is to optimize the prediction of GMV and provide insights that can guide business decisions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Combining DataFrames](#combining-dataframes)
5. [Methodology](#methodology)
6. [Evaluation Metrics](#evaluation-metrics)


## Project Overview

In this project, we perform the following steps:

- **Data Preparation**: Handling missing values, combining multiple dataframes, dropping irrelevant columns, and splitting the data into training and testing sets.
- **EDA (Exploratory Data Analysis)**: Understanding data distributions, correlations, and feature importance.
- **Modeling**: Building a basic **Linear Regression model** and refining it using **Recursive Feature Elimination with Cross-Validation (RFECV)**.
- **Feature Selection**: Using RFECV to select the most relevant features, improving model performance and interpretability.
- **Model Evaluation**: Assessing model performance using metrics like **R² Score**, **RMSE**, **MSE**, and **MAE**.

## Dataset

The dataset consists of order-level information with the following key columns:
- `fsn_id`: Product identifier
- `order_date`: Date of the order
- `order_id`: Unique order identifier
- `gmv`: Gross Merchandise Value (target variable)
- `units`: Number of units ordered
- `deliverybdays`: Business days for delivery
- `cust_id`: Customer identifier
- `product_analytic_sub_category`: Product subcategory
- ...and more.

The dataset contains a total of 20 columns and 1.6 million rows.

## Exploratory Data Analysis (EDA)

Before building the machine learning model, we conduct an in-depth **Exploratory Data Analysis (EDA)** to understand the dataset. This includes:

- **Summary Statistics**: Viewing summary statistics of key variables to understand distributions, means, medians, and ranges.
- **Correlation Analysis**: Using correlation matrices to assess relationships between numerical features and the target variable (`gmv`).
- **Feature Distributions**: Plotting distributions for key features to understand their spread and check for any outliers.
- **Visualization**: Using **Matplotlib** and **Seaborn** for plotting histograms, box plots, heatmaps, and pair plots to visualize data patterns and feature relationships.

## Combining DataFrames

If the data is split into multiple tables or sources, they need to be combined appropriately:

- **Merging DataFrames**: Using techniques such as `pd.merge()` or `join()` to combine datasets on common keys (e.g., `order_id`, `cust_id`).
- **Handling Duplicates**: Checking for duplicate entries and removing them to ensure data integrity.
- **Handling Missing Values**: After merging, checking for any null values and handling them either by imputation or removal.

## Methodology

1. **Check for Null Values**:
   - Begin by checking the dataset for missing values (nulls) in all columns.
   - Handle missing data by imputing or removing rows/columns as necessary.

2. **Drop the Week Column**:
   - Remove the `Week` column since it is a row identifier and does not contribute to the prediction of revenue (GMV).

3. **Train-Test Split**:
   - **Target Variable**: The target variable for prediction is `gmv` (Gross Merchandise Value).
   - **Features**: Use all other columns (except `Week`) as features.
   - Split the dataset into training and testing sets, typically using an 80/20 or 70/30 ratio for training and testing.

4. **Apply Standard Scaler**:
   - Standardize the features using **StandardScaler** to ensure all features are on the same scale, especially important for linear regression.

---

### Model Building

5. **Train Basic Linear Regression Model**:
   - Fit a basic **Linear Regression model** using the training data (scaled features and target `gmv`).

6. **Evaluate the Model's Metrics**:
   - Calculate and evaluate common regression metrics such as:
     - **R² Score**: To assess the goodness of fit.
     - **Root Mean Squared Error (RMSE)**: To measure the average error magnitude.
     - **Mean Squared Error (MSE)**: To quantify the squared differences between predicted and actual values.
     - **Mean Absolute Error (MAE)**: To assess the absolute differences between predicted and actual values.

---

### Feature Selection and Optimization

7. **Feature Selection using Recursive Feature Elimination with Cross-Validation (RFECV)**:
   - Use **RFECV** to identify the most important features by recursively eliminating less important features.
   - Perform cross-validation during the feature elimination process to ensure the optimal subset of features is selected.

8. **Evaluate Accuracy with Selected Features**:
   - Assess the performance of the model using only the features selected by **RFECV**.
   - Calculate the same evaluation metrics (R², RMSE, MSE, MAE) to compare with the initial model.

9. **Build a New Model Using RFECV-Selected Features**:
   - Train a new **Linear Regression model** using only the features identified by **RFECV**.

10. **Evaluate the New Model**:
   - Evaluate the new model using the same metrics (R², RMSE, MSE, MAE).
   - Compare the performance of this refined model with the original model to determine the improvement brought by feature selection.

---

## Evaluation Metrics

We evaluate the models using the following metrics:
- **R² Score**: Measures the proportion of variance explained by the model.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared differences between actual and predicted values.
- **Mean Squared Error (MSE)**: Measures the average of the squares of the differences between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average of the absolute differences between actual and predicted values.


