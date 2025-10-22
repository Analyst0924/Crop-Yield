# Crop Yield Prediction and Analysis using Machine Learning

## Project Overview

This project provides an end-to-end data science solution for analyzing and predicting agricultural crop yield. Using a comprehensive dataset of environmental and farming factors, we developed a suite of machine learning models to tackle the problem from multiple angles: classification, regression, and clustering.

The primary goal was to not only build accurate predictive models but also to derive actionable insights that could inform agricultural decision-making. The analysis identified key drivers of yield, uncovered hidden farming patterns, and established a robust methodology for evaluating model performance. The final classification model, a Random Forest Classifier, achieved an outstanding **98.6% accuracy** and an **ROC AUC of 0.997**.

### Technical Stack
*   **Languages:** Python
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Project Workflow

1.  **Data Preparation & Exploratory Data Analysis (EDA)**
    *   Loaded and cleaned the dataset, confirming data quality with no missing values.
    *   Performed EDA using histograms and pairplots to understand feature distributions, identifying a heavy skew in pesticide usage and a wide variance in crop yield.
    *   Utilized `LabelEncoder` for categorical features and `StandardScaler` for numerical features to prepare the data for modeling.

2.  **Classification Modeling: High vs. Low Yield**
    *   Engineered a binary target variable by classifying yield as "High" (>= median) or "Low" (< median).
    *   Implemented and compared two models: **K-Nearest Neighbors (KNN)** and **Random Forest Classifier**.
    *   The Random Forest model significantly outperformed KNN, achieving **98.6% accuracy** and an **ROC AUC of 1.00** on the test set, demonstrating its robustness and strong generalization capabilities.
    *   Optimized the Random Forest model using `GridSearchCV` to tune hyperparameters (`n_estimators`, `max_depth`), ensuring peak performance.

3.  **Regression Modeling: Predicting Continuous Yield**
    *   To predict the exact crop yield value (hg/ha), we implemented and evaluated several regression models:
        *   Linear Regression
        *   Lasso and Ridge Regression (for feature selection)
        *   ElasticNet
        *   Decision Tree Regressor
    *   Evaluated models based on Mean Squared Error (MSE) and R² Score, with the Decision Tree Regressor proving most effective at capturing non-linear relationships.

4.  **Unsupervised Learning: Identifying Yield Patterns**
    *   Applied **K-Means Clustering** to segment the data and uncover hidden patterns in crop environments.
    *   Used the Elbow Method and Silhouette Analysis to determine the optimal number of clusters (k=10).
    *   Visualized clusters using PCA, revealing distinct groups such as:
        *   **Cluster 9 (High Efficiency):** High yield (~106,000 hg/ha) with the lowest pesticide usage.
        *   **Cluster 2 (Inefficiency):** High pesticide usage (~294,000 tonnes) for only a moderate yield.

5.  **Dimensionality Reduction & Feature Importance**
    *   Evaluated the impact of **Principal Component Analysis (PCA)** on model performance. Retaining 95% of variance with PCA maintained high accuracy (~96.5%) for the Random Forest model, presenting a viable option for larger datasets.
    *   Used **Lasso and Ridge regression coefficients** to identify crop type as the most influential feature, with potatoes showing a strong positive impact on yield.

## Key Results and Insights

*   **Best Classification Model:** The tuned Random Forest Classifier is the superior model for predicting high vs. low yield categories, with near-perfect performance metrics.
*   **Actionable Clustering Insights:** K-Means clustering successfully identified distinct farming patterns, providing data-driven evidence for optimizing pesticide usage and improving efficiency.
*   **Key Yield Drivers:** Crop type is the most critical predictor of yield. Environmental factors like rainfall and temperature also play a significant role.
*   **Final Recommendation:** For a more complex, nuanced prediction task, **Gradient Boosting** was recommended, as its iterative learning process is well-suited for capturing subtle patterns in the data, achieving the highest accuracy (62.9%) in a final comparative analysis.

## Team
*   Ashraf Afana
*   Anu Girija
*   Amanjot Kandhola
*   Alan Mathews
*   Anishka Shetty
