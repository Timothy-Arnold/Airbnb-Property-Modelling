import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd

import tabular_data
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import 

# Product features and labels directly from the csv
clean_data = pd.read_csv("clean_tabular_data.csv")
features = clean_data.select_dtypes(include = ["int64", "float64"])
label_series = clean_data["Category"]

# Encode labels

label_categories = label_series.unique()
le = LabelEncoder()
label_encoded = le.fit_transform(label_series)

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

def print_performance(y_hat_train, y_hat_validation):
    train_precision = precision_score(y_train, y_hat_train, average="micro", zero_division=0)
    train_recall = recall_score(y_train, y_hat_train, average="micro")
    train_f1 = f1_score(y_train, y_hat_train, average="micro")
    test_precision = precision_score(y_validation, y_hat_validation, average="micro", zero_division=0)
    test_recall = recall_score(y_validation, y_hat_validation, average="micro")
    test_f1 = f1_score(y_validation, y_hat_validation, average="micro")
    print("Train precision", train_precision)
    print("Train recall", train_recall)
    print("Train f1", train_f1)
    print("Validation precision", test_precision)
    print("Validation recall", test_recall)
    print("Validation f1", test_f1)

if  __name__ == '__main__':
    np.random.seed(2)
    y_hat = create_first_model()
    print_performance(y_hat[0], y_hat[1])