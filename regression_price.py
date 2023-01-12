import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd

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

#Split dataset
clean_data = pd.read_csv("clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
# print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5
)

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
    best_model_list = [best_model.fit(X_train, y_train), GS.best_params_, performance_metrics_dict]
    print(best_model_list)
    return best_model_list

def save_model(model_list, folder="models/regression_price/linear_regression"):
    model = model_list[0]
    hyper_params = model_list[1]
    performance_metrics = model_list[2]
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, f"{folder}/model.joblib")
    with open(f"{folder}/hyperparameters.json", 'w') as fp:
        json.dump(hyper_params, fp)
    with open(f"{folder}/metrics.json", 'w') as fp:
        json.dump(performance_metrics, fp)

def evaluate_all_models(task_folder="models/regression_price"):
    np.random.seed(2)
    decision_tree_model = tune_regression_model_hyperparameters("DecisionTreeRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [15, 30, 45, 60],
    "min_samples_split": [2, 4, 0.2, 0.4],
    "max_features": [4, 6, 8]
    })

    save_model(decision_tree_model, folder=f"{task_folder}/decision_tree")

    random_forest_model = tune_regression_model_hyperparameters("RandomForestRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [50, 100, 150],
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [30, 40, 50],
    "min_samples_split": [2, 0.1, 0.2],
    "max_features": [1, 2]
    })

    save_model(random_forest_model, folder=f"{task_folder}/random_forest")

    gradient_boosting_model = tune_regression_model_hyperparameters("GradientBoostingRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [25, 50, 100],
    "loss": ["squared_error", "absolute_error"],
    "max_depth": [1, 3, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_features": [1, 2, 3]
    })

    save_model(gradient_boosting_model, folder=f"{task_folder}/gradient_boosting")

    return decision_tree_model, random_forest_model, gradient_boosting_model

def find_best_model(model_details_list):
    validation_scores = [x[2]["validation_RMSE"] for x in model_details_list]
    best_score_index = np.argmin(validation_scores)
    best_model_details = model_details_list[best_score_index]
    return best_model_details

if  __name__ == '__main__':
    np.random.seed(2)
    model_details_list = evaluate_all_models()
    best_model_details = find_best_model(model_details_list)
    print(f"The best model: {best_model_details}")