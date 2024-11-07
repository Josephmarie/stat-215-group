# import packages and load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import os
from pyreadr import read_r
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import joblib

image_1 = pd.read_csv("image1.txt", delim_whitespace=True, header=None)
image_2 = pd.read_csv("image2.txt", delim_whitespace=True, header=None)
image_3 = pd.read_csv("image3.txt", delim_whitespace=True, header=None)

column_names = ['y_coor', 'x_coor', 'expert_label', 'NDAI', 'SD', 'CORR', 'Radiance_angle_DF','Radiance_angle_CF','Radiance_angle_BF','Radiance_angle_AF', 'Radiance_angle_AN'] 
image_1.columns = column_names
image_2.columns = column_names
image_3.columns = column_names

# remove when expert label is zero (remove uncertain data)
train_on2 = image_2[image_2['expert_label'] != 0]

# select all features except the expert_label column from dataset and call it x
x_train2 = train_on2.drop(columns=['expert_label']) # explanatory variables 
y_train2 = train_on2['expert_label'] # response variable we want to predict

# below are datasets we should use for caculating prediction accuracy using image 2
validate_on1 = image_1[image_1['expert_label'] != 0] # remove uncertain data


# Try to tune the hyperparameters using gridsearch and cross validation (train on 1 and validate on 2)
rf = RandomForestClassifier(random_state=42)
param_grid = {
'n_estimators': [100, 200, 300], # number of trees in the forest, default=100. usually more trees better performance
'max_depth': [None, 10, 20, 30], # the maximum depth of the tree, default=none. may lead to overfitting if the depth too large
'min_samples_split': [2, 5, 10], # the minimum number of samples required to split an internal node, default=2 (the node will split as long as it has at least 2 samples). high values prevent overfitting
'min_samples_leaf': [1, 2, 4], # the minimum number of samples required to be at a leaf node, default=1. large values prevent overfitting
'criterion': ['gini', 'entropy'], # the function to measure the quality of a split, default=gini. 
'bootstrap': [True, False] # whether bootstrap samples are used when building trees, default=True
}


# Create a validation fold
X = pd.concat([train_on2.drop(columns=['expert_label']), validate_on1.drop(columns=['expert_label'])], ignore_index=True)
y = pd.concat([train_on2['expert_label'], validate_on1['expert_label']], ignore_index=True)

# Create a validation fold: -1 for training data on image 1, 0 for validation data on image 2
test_fold = [-1] * len(train_on2) + [0] * len(validate_on1)
ps = PredefinedSplit(test_fold)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=ps, n_jobs=26)

#fit the model 
grid_search.fit(X, y)

best_score = grid_search.best_score_
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

# Create a dictionary to hold the best parameters and score
results = {
    'best_score': [best_score],
    'best_params': [best_params]
}

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv('grid_search_results2.csv', index=False)

# Save the best model
joblib.dump(best_rf, 'best_random_forest_model2.pkl')

