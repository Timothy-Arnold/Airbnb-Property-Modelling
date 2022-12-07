import numpy as np
import pandas as pd

import tabular_data
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve

np.random.seed(2)

#Split dataset
clean_data = pd.read_csv("clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5
)

#Fit to model
sgd = SGDRegressor()
sgd.fit(X_train, y_train)
y_hat_train = sgd.predict(X_train)
y_hat_test = sgd.predict(X_test)

def print_errors():
    train_rmse = mean_squared_error(y_train, y_hat_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_hat_test, squared=False)
    test_mae = mean_absolute_error(y_test, y_hat_test)
    test_mse = mean_squared_error(y_test, y_hat_test)
    train_r2 = r2_score(y_train, y_hat_train)
    test_r2 = r2_score(y_test, y_hat_test)

    print("Mean root squared error on Training set: ", train_rmse)
    print("Mean root squared error on Test set: ", test_rmse)
    print("Mean absolute error on Test set: ", test_mae)
    print("Mean squared error on Test set: ", test_mse)
    print("R squared on Train set: ", train_r2)
    print("R squared on Test set: ", test_r2)

def custom_tune_regression_model_hyperparameters():
    pass

if __name__ == '__main__':
    print_errors()