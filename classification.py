import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd

import tabular_data
# from sklearn.ensemble import 
# from sklearn.ensemble import 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import 

# Product features and labels directly from the csv
clean_data = pd.read_csv("clean_tabular_data.csv")
features = clean_data.select_dtypes(include = ["int64", "float64"])
label_series = clean_data["Category"]
print(label_series.tail(30))

# Encode labels

label_categories = label_series.unique()
print(label_categories)
le = LabelEncoder()
label_encoded = le.fit_transform(label_series)
print(label_encoded[-30:])

X, y = (features, label_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5
)

def create_first_model():
    log_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=100) 
    log_reg.fit(X_train, y_train)
    y_hat_train = log_reg.predict(X_train)
    y_hat_validation = log_reg.predict(X_validation)
    return [y_hat_train, y_hat_validation]

def print_errors(y_hat_train, y_hat_validation):
    train_rmse = mean_squared_error(y_train, y_hat_train, squared=False)
    validation_rmse = mean_squared_error(y_validation, y_hat_validation, squared=False)
    validation_mae = mean_absolute_error(y_validation, y_hat_validation)
    validation_mse = mean_squared_error(y_validation, y_hat_validation)
    train_r2 = r2_score(y_train, y_hat_train)
    validation_r2 = r2_score(y_validation, y_hat_validation)

    print("Mean root squared error on Training set: ", train_rmse)
    print("Mean root squared error on Validation set: ", validation_rmse)
    print("Mean absolute error on Validation set: ", validation_mae)
    print("Mean squared error on Validation set: ", validation_mse)
    print("R squared on Train set: ", train_r2)
    print("R squared on Validation set: ", validation_r2)

if  __name__ == '__main__':
    np.random.seed(2)
    y_hat = create_first_model()
    print_errors(y_hat[0], y_hat[1])
