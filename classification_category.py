import collections
import joblib
import json
import numpy as np
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

np.random.seed(2)

hyper_param_dict = {
    "LogisticRegression": {
        "tol": [1E-5, 1E-4, 1E-3],
        "max_iter": [100, 500, 1000],
        "multi_class": ["multinomial"]
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": [15, 30, 45, 60],
        "min_samples_split": [2, 4, 0.2, 0.4],
        "max_features": [4, 6, 8]
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "criterion": ["gini", "entropy"],
        "max_depth": [30, 40, 50],
        "min_samples_split": [2, 0.1, 0.2],
        "max_features": [1, 2, 3]
    },
    "GradientBoostingClassifier": {
        "n_estimators": [25, 50, 100],
        "loss": ["log_loss"],
        "max_depth": [1, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_features": [1, 2, 3]
    }
}

model_class_list = hyper_param_dict.keys()

model_folder_name_dict = {
                        "LogisticRegression" : "logistic_regression",
                        "DecisionTreeClassifier" : "decision_tree", 
                        "RandomForestClassifier" : "random_forest",
                        "GradientBoostingClassifier" : "gradient_boosting"
                        }

def create_first_model():
    log_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=100) 
    log_reg.fit(X_train, y_train)
    y_hat_train = log_reg.predict(X_train)
    y_hat_validation = log_reg.predict(X_validation)
    return [y_hat_train, y_hat_validation]

def print_performance(y_hat_train, y_hat_validation):
    train_precision = precision_score(y_train, y_hat_train, average="macro", zero_division=0)
    train_recall = recall_score(y_train, y_hat_train, average="macro")
    train_f1 = f1_score(y_train, y_hat_train, average="macro")
    train_report = classification_report(y_train, y_hat_train)
    test_precision = precision_score(y_validation, y_hat_validation, average="macro", zero_division=0)
    test_recall = recall_score(y_validation, y_hat_validation, average="macro")
    test_f1 = f1_score(y_validation, y_hat_validation, average="macro")
    test_report = classification_report(y_validation, y_hat_validation)
    print("Train precision", train_precision)
    print("Train recall", train_recall)
    print("Train f1", train_f1)
    print("Train report", train_report)
    print("Validation precision", test_precision)
    print("Validation recall", test_recall)
    print("Validation f1", test_f1)
    print("Validation report", test_report)

def tune_classification_model_hyperparameters(model_class, 
    X_train, y_train, X_validation, y_validation, search_space):
    models_list =   {
                    "LogisticRegression" : LogisticRegression,
                    "DecisionTreeClassifier" : DecisionTreeClassifier, 
                    "RandomForestClassifier" : RandomForestClassifier,
                    "GradientBoostingClassifier" : GradientBoostingClassifier
                    }
    model = models_list[model_class]()
    GS = GridSearchCV(estimator = model, 
                      param_grid = search_space, 
                      scoring = "accuracy",
                      )
    GS.fit(X_train, y_train)
    best_model = models_list[model_class](**GS.best_params_)
    best_model.fit(X_train, y_train)
    y_hat_validation = best_model.predict(X_validation)
    validation_accuracy = accuracy_score(y_validation, y_hat_validation)
    validation_f1 = f1_score(y_validation, y_hat_validation, average="macro")
    performance_metrics_dict = {"validation_accuracy": validation_accuracy, "validation_f1": validation_f1}
    best_model_details = [best_model.fit(X_train, y_train), GS.best_params_, performance_metrics_dict]
    print(best_model_details)
    return best_model_details

def save_model(model_details, folder="models/classification_category/logistic_regression"):
    model = model_details[0]
    hyper_params = model_details[1]
    performance_metrics = model_details[2]
    if not os.path.exists(folder):
        os.makedirs(folder)

    joblib.dump(model, f"{folder}/model.joblib")
    with open(f"{folder}/hyperparameters.json", 'w') as fp:
        json.dump(hyper_params, fp)
    with open(f"{folder}/metrics.json", 'w') as fp:
        json.dump(performance_metrics, fp)

def evaluate_all_models(task_folder="models/classification_category"):
    #Initialize dictionary of models
    model_details_dict = {}
    for model_class in model_class_list:

        model_details_dict[f"{model_class}"] = tune_classification_model_hyperparameters(
            model_class,
            X_train,
            y_train,
            X_validation,
            y_validation,
            search_space = hyper_param_dict[f"{model_class}"]
        )

        model_folder_name = model_folder_name_dict[f"{model_class}"]
        save_model(model_details_dict[f"{model_class}"], folder=f"{task_folder}/{model_folder_name}")
    return model_details_dict

def find_best_model(model_details_list):
    # Initialize validation accuracy to be maximized
    highest_accuracy_validation = 0
    for model_class in model_class_list:
        model_details = model_details_list[model_class]
        accuracy_validation = model_details[2]["validation_accuracy"]
        if accuracy_validation > highest_accuracy_validation:
            highest_accuracy_validation = accuracy_validation
            best_model_details = model_details
    return best_model_details

if  __name__ == '__main__':
    clean_data = pd.read_csv("clean_tabular_data.csv")
    features = clean_data.select_dtypes(include = ["int64", "float64"])
    label_series = clean_data["Category"]
    # Encode labels
    label_categories = label_series.unique()
    le = LabelEncoder()
    label_encoded = le.fit_transform(label_series)

    # Count how many of each label there is
    label_count = collections.Counter(label_encoded)
    for key, value in label_count.items():
        print(f"{key}: {value}")

    X, y = (features, label_encoded)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5
    )
    y_hat = create_first_model()
    # print_performance(y_hat[0], y_hat[1])
    model_details_list = evaluate_all_models()
    best_model_details = find_best_model(model_details_list)
    print(f"The best model: {best_model_details}")