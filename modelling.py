import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd

import tabular_data
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from torchvision.models import resnet50, resnet18

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
        models_list = {"ResNet-50" : resnet50, "ResNet-18" : resnet18, "SGDRegression" : SGDRegressor}
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
    models_list = {"ResNet-50" : resnet50, "ResNet-18" : resnet18, "SGDRegressor" : SGDRegressor, "DecisionTreeRegressor" : DecisionTreeRegressor, "RandomForestRegressor" : RandomForestRegressor}
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
    print(GS.best_score_)
    best_model_list = [best_model.fit(X_train, y_train), GS.best_params_, performance_metrics_dict]
    print(best_model_list)
    return best_model_list

def save_model(model_list, folder="models/regression/linear_regression"):
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

if  __name__ == '__main__':
    np.random.seed(2)
    # best_model_list = custom_tune_regression_model_hyperparameters("SGDRegression", 
    # X_train, y_train, X_validation, y_validation, X_test, y_test, 
    # search_space = {
    # "penalty": ["l1", "l2", "elasticnet"],
    # "early_stopping": [True, False], 
    # "learning_rate": ["constant", "invscaling", "adaptive"], 
    # "max_iter": [500, 1000, 1500, 2000]
    # })
    linear_regression_model = tune_regression_model_hyperparameters("SGDRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "penalty": ["l1", "l2", "elasticnet"],
    "early_stopping": [True, False], 
    "learning_rate": ["constant", "invscaling", "adaptive"], 
    "max_iter": [500, 1000, 1500, 2000]
    })
    save_model(linear_regression_model)

    decision_tree_model = tune_regression_model_hyperparameters("DecisionTreeRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [15, 30, 45, 60],
    "min_samples_split": [2, 4, 0.2, 0.4], 
    "max_features": [8, 6, 4]
    })
    save_model(decision_tree_model, folder="models/regression/decision_tree")

