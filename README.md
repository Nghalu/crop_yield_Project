# Crop Yield Prediction Using Machine Learning
##Project Overview
This project uses machine learning to predict crop yield based on environmental and agricultural factors such as rainfall, temperature, and pesticide usage.
The work supports UN Sustainable Development Goal 2: Zero Hunger, by helping optimize agricultural productivity and resource use through data-driven insights.
## Dataset Description
### The dataset (from Kaggle) consists of multiple CSV files:
- pesticides.csv — Amount of pesticides used (tonnes or kg/ha)
- rainfall.csv — Average rainfall per country or region (mm)
- temp.csv — Average temperature data (°C)
- yield.csv — Crop yield measurements (hg/ha)
- yield1.csv — Additional or cleaned yield data (depending on dataset structure)
### Each file contains yearly records by country or region and can be merged on common columns such as Year, Area, or Country.
## Technologies Used
Python 3.12+
pandas, numpy
matplotlib, seaborn
scikit-learn
LightGBM, XGBoost
## Project Steps
- Data Loading & Inspection
- Loading CSV files and explore data structures, missing values, and data types.
- Data Cleaning & Preprocessing
- Handle missing values, inconsistent data, and outliers.
- Convert units and ensure numeric types.
- Exploratory Data Analysis (EDA)
- Visualize distributions, correlations, and relationships between features.
- Understand which factors affect crop yield most.
- Feature Engineering
- Creating new features such as rainfall-to-temperature ratios or lagged values.
- ### Modeling
- Use Random Forest Regressor and Gradient Boosting (LightGBM/XGBoost) to predict yield.
- Evaluate models using R² score and RMSE.
