# ML-Project-Personal-Key-Indicators-of-Heart-Disease
Project for CS 6375 

Dataset:
# Personal Key Indicators of Heart Disease

**Authors:** Shreeprasad Anant Sonar, Abhishek Chauhan, Aniket Kulkarni

**Date:** May 8, 2023

In this project, we utilized the 2020 annual CDC survey data of 400K US adults to create models that predict the chance of developing heart disease. Our dataset included important markers such as high blood pressure, high cholesterol, smoking, diabetes, obesity, physical activity, and alcohol use. The dataset was preprocessed by cleaning, normalization, and converting the "HeartDisease" variable to a binary class.

## Project Objective

The main goal of our project was to identify and reduce the risk factors associated with heart disease. We aimed to predict heart disease using the collected data from the annual CDC survey. With the help of machine learning algorithms, we aimed to create models that can anticipate heart disease by forecasting the chances of developing the disease. 

## Dataset:  https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

The dataset was provided by the Behavioral Risk Factor Surveillance System (BRFSS) and included health status information of 400K US individuals related to heart disease risk factors such as high blood pressure, high cholesterol, smoking, diabetes status, obesity, physical activity, and alcohol use. The "HeartDisease" variable was treated as binary, and classes were not balanced.

## Exploratory Data Analysis

The dataset was found to be imbalanced, with the number of respondents with heart disease much lower than those who do not have heart disease. The distribution of the "Sex" variable was found to be balanced for the "No" label, with 54% of respondents being female and 46% being male. For the "Yes" label, there were 41% female respondents and 59% male respondents.

We also performed correlation analysis between variables to determine the strongest correlations and to plot graphs to depict the links between variables with the highest correlation values.

## Machine Learning Methods

We compared custom implementations of four classifiers, namely Logistic Regression, Naive Bayes, Decision Tree along with Bagging, AdaBoost, and XGBoost, with the corresponding Scikit-learn models under 10 Fold cross-validation and sampling the data in ratios from 1:1 to 9:1 to analyze the accuracies, precision, recall, and F1 scores.

## Conclusion

We were able to build machine learning models that can forecast the chance of developing heart disease with good accuracy, precision, recall, and F1 scores. Our findings can provide insights into the trends and causes of heart disease, which can help in identifying and reducing the risk factors associated with it. The project can be extended further by including more features and data from other sources to enhance the accuracy of the models.
