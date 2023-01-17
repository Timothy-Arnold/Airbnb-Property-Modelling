import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

import tabular_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

np.random.seed(2)

hyper_param_dict = {
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [15, 30, 45, 60],
        "min_samples_split": [2, 4, 0.2, 0.4],
        "max_features": [6, 10, 14]
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 150],
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [40, 50, 60],
        "min_samples_split": [2, 0.1, 0.2],
        "max_features": [2, 4, 6]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [25, 50, 100],
        "loss": ["squared_error", "absolute_error"],
        "max_depth": [1, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_features": [1, 3, 5]
    }
}

model_class_list = hyper_param_dict.keys()

model_folder_name_dict = {
                        "SGDRegressor" : "linear_regression",
                        "DecisionTreeRegressor" : "decision_tree", 
                        "RandomForestRegressor" : "random_forest",
                        "GradientBoostingRegressor" : "gradient_boosting"
                        }

def create_first_model():
    sgd = SGDRegressor() 
    sgd.fit(X_train, y_train)
    y_hat_train = sgd.predict(X_train)
    y_hat_validation = sgd.predict(X_validation)
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

def custom_tune_regression_model_hyperparameters(model_class, 
    X_train, y_train, X_validation, y_validation, X_test, y_test, 
    search_space):
    fitted_model_list = []
    keys = search_space.keys()
    vals = search_space.values()
    for instance in itertools.product(*vals):
        hyper_values_dict = dict(zip(keys, instance))
        models_list =   {
                        "SGDRegressor" : SGDRegressor,
                        "DecisionTreeRegressor" : DecisionTreeRegressor, 
                        "RandomForestRegressor" : RandomForestRegressor,
                        "GradientBoostingRegressor" : GradientBoostingRegressor
                        }
        model = models_list[model_class](**hyper_values_dict)
        model.fit(X_train, y_train)
        y_hat_validation = model.predict(X_validation)
        validation_rmse = mean_squared_error(y_validation, y_hat_validation, squared=False)
        validation_r2 = r2_score(y_validation, y_hat_validation)
        performance_metrics_dict = {"validation_RMSE": validation_rmse, "validation_R2": validation_r2}
        model_details = [model, hyper_values_dict, performance_metrics_dict]
        fitted_model_list.append(model_details)
    best_model_list = min(fitted_model_list, key=lambda x: x[2]["validation_RMSE"])
    print(f"Best model by validation_RMSE metric:\n{best_model_list}")
    return best_model_list

def tune_regression_model_hyperparameters(model_class, 
    X_train, y_train, X_validation, y_validation, search_space):
    models_list =   {
                    "SGDRegressor" : SGDRegressor,
                    "DecisionTreeRegressor" : DecisionTreeRegressor, 
                    "RandomForestRegressor" : RandomForestRegressor,
                    "GradientBoostingRegressor" : GradientBoostingRegressor
                    }
    model = models_list[model_class]()
    GS = GridSearchCV(estimator = model, 
                      param_grid = search_space, 
                      scoring = ["r2", "neg_root_mean_squared_error"], 
                      refit = "neg_root_mean_squared_error"
                      )
    GS.fit(X_train, y_train)
    best_model = models_list[model_class](**GS.best_params_)
    best_model.fit(X_train, y_train)
    y_hat_validation = best_model.predict(X_validation)
    validation_rmse = mean_squared_error(y_validation, y_hat_validation, squared=False)
    validation_r2 = r2_score(y_validation, y_hat_validation)
    performance_metrics_dict = {"validation_RMSE": validation_rmse, "validation_R2": validation_r2}
    best_model_details = [best_model.fit(X_train, y_train), GS.best_params_, performance_metrics_dict]
    print(best_model_details)
    return best_model_details

def save_model(model_details, folder="models/regression_bedrooms/linear_regression"):
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

def evaluate_all_models(task_folder="models/regression_bedrooms"):
    #Initialize dictionary of models
    model_details_dict = {}
    for model_class in model_class_list:

        model_details_dict[f"{model_class}"] = tune_regression_model_hyperparameters(
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
    # Initialize RMSE loss to be minimized
    lowest_RMSE_loss_validation = np.inf
    for model_class in model_class_list:
        model_details = model_details_list[model_class]
        RMSE_loss_validation = model_details[2]["validation_RMSE"]
        if RMSE_loss_validation < lowest_RMSE_loss_validation:
            lowest_RMSE_loss_validation = RMSE_loss_validation
            best_model_details = model_details
    return best_model_details

if  __name__ == '__main__':
    clean_data = pd.read_csv("clean_tabular_data.csv")
    X, y = tabular_data.load_airbnb(clean_data, "bedrooms")
    category_series = clean_data["Category"]
    category_options = category_series.unique()
    one_hot = pd.get_dummies(category_series)
    one_hot = one_hot.astype("int64")
    X = pd.concat([X, one_hot], axis=1)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5
    )
    model_details_list = evaluate_all_models()
    best_model_details = find_best_model(model_details_list)
    print(f"The best model: {best_model_details}")