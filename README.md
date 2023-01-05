# Airbnb-Property-Modelling

I decided to use an Ubuntu VM for this project in order to practice coding on Linux.

## Milestone 1

First of all I had to clean my text and image data to prepare it for use in my models:

For the text data the Descriptions came in lists of phrases, so I had to turn them into single long strings. I also removed rows with missing Value Ratings, as well as filled in the blanks for rows with missing values for guests, beds, bathrooms or bedrooms. I then filtered out only the columns with numerical data (for the sake of future modelling), and made a function to return one of these columns (as the label) and the rest in a data frame (as the features).

```python

import numpy as np
import pandas as pd
from ast import literal_eval

def remove_rows_with_missing_ratings(raw_data):
    raw_data.dropna(subset=["Value_rating"], inplace=True)
    raw_data.drop(raw_data.columns[-1], axis = 1, inplace=True)
    return raw_data

def combine_description_strings(desc):
    desc_list = literal_eval(desc.strip())[1:]
    desc_list_without_empty_quotes = list(filter(lambda x: x != "", desc_list))
    full_desc = ' '.join(desc_list_without_empty_quotes)
    return full_desc

def is_valid_description(description):
    try: 
        description_type = type(literal_eval(description))
        if description_type == list:
            return description
        else:
            return np.nan
    except:
        return np.nan

def fix_description_strings(df):
    df['Description'] = df['Description'].apply(is_valid_description)
    df.dropna(subset=["Description"], inplace=True)
    df["Description"] = df["Description"].apply(combine_description_strings)
    return df

def set_default_feature_values(df):
    df[["guests", "beds", "bathrooms", "bedrooms"]] = df[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1)
    return df

def clean_tabular_data(df):
    raw_data_with_ratings = remove_rows_with_missing_ratings(df)
    raw_data_with_description = fix_description_strings(raw_data_with_ratings)
    raw_data_default_features = set_default_feature_values(raw_data_with_description)
    return raw_data_default_features

def load_airbnb(df, label):
    df_numerical = df.select_dtypes(include = ["int64", "float64"])
    print(df_numerical.info())
    print(df_numerical.loc[:, "bathrooms"].head(5))
    label_series = df_numerical[label]
    features = df_numerical.drop(label, axis=1)
    full_data = (features, label_series)
    return full_data

if __name__ == '__main__':
    raw_data = pd.read_csv("listing.csv", index_col=0)
    clean_data = clean_tabular_data(raw_data)
    clean_data.to_csv("clean_tabular_data.csv")
    full_data = load_airbnb(clean_data, "bathrooms")
```

For the image data I found the minimum height of all the images, resized them all to have the same height (while keeping their aspect ratios), and downloaded all of these smaller images in my local directory for future models.

```python
from PIL import Image
import glob
import os

class PrepareImages:

    def __init__(self):
        self.image_name_list = []
        self.image_list = []
        self.height_list = []
        self.resized_image_list = []

    def __import_images(self):
        for filename in glob.glob("/home/timothy/Documents/Aicore/Airbnb-Property-Modelling/images/*/*.png"):
            print(filename)
            img = Image.open(filename)
            if img.mode == "RGB":
                img_name = filename[-42:]
                self.image_name_list.append(img_name)
                self.image_list.append(img)
                self.height_list.append(img.height)
        print(f"There are {len(self.image_list)} RGB images")

    def __resize_images(self):
        minimum_height = min(self.height_list)
        print(f"The minimum height is {minimum_height}")
        for img in self.image_list:
            aspect_ratio = img.width / img.height
            new_width = int(minimum_height * aspect_ratio)
            new_size = (new_width, minimum_height)
            resized_image = img.resize(new_size)
            self.resized_image_list.append(resized_image)
        print("All images resized!")

    def __download_resized_images(self):
        for img_name, img in zip(self.image_name_list, self.resized_image_list):
            file_path = os.path.join("/home/timothy/Documents/Aicore/Airbnb-Property-Modelling/processed_images", img_name)
            img.save(file_path)
        print("All resized images downloaded")

    def do_whole_resize(self):
        PrepareImages.__import_images(self)
        PrepareImages.__resize_images(self)
        PrepareImages.__download_resized_images(self)

if __name__ == '__main__':
    resize = PrepareImages()
    resize.do_whole_resize()
```

## Milestone 2

I chose 4 regression models to try and model my data with, using sci-kit learn to fit them. I used GridsearchCV to perform grid searches on each of them with a bunch of different hyperparameter options. I also made my own custom gridsearch function for practice.

I saved the best models, along with their hyperparameters and metric scores in a folder.

```python

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

def evaluate_all_models():
    np.random.seed(2)
    decision_tree_model = tune_regression_model_hyperparameters("DecisionTreeRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [15, 30, 45, 60],
    "min_samples_split": [2, 4, 0.2, 0.4],
    "max_features": [4, 6, 8]
    })

    save_model(decision_tree_model, folder="models/regression/decision_tree")

    random_forest_model = tune_regression_model_hyperparameters("RandomForestRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [50, 100, 150],
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [30, 40, 50],
    "min_samples_split": [2, 0.1, 0.2],
    "max_features": [1, 2]
    })

    save_model(random_forest_model, folder="models/regression/random_forest")

    gradient_boosting_model = tune_regression_model_hyperparameters("GradientBoostingRegressor", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [25, 50, 100],
    "loss": ["squared_error", "absolute_error"],
    "max_depth": [1, 3, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_features": [1, 2, 3]
    })

    save_model(gradient_boosting_model, folder="models/regression/gradient_boosting")

    return decision_tree_model, random_forest_model, gradient_boosting_model

def find_best_model(model_details_list):
    validation_scores = [x[2]["validation_RMSE"] for x in model_details_list]
    print(validation_scores)
    best_score_index = np.argmin(validation_scores)
    print(best_score_index)
    best_model_details = model_details_list[best_score_index]
    return best_model_details

if  __name__ == '__main__':
    np.random.seed(2)
    model_details_list = evaluate_all_models()
    best_model_details = find_best_model(model_details_list)
    print(f"The best model: {best_model_details}")
```

The metrics of the 4 models I created:

Linear Regression: `{"validation_RMSE": 2903929721.9552526, "validation_R2": -598955739200091.6}` (was just to form a baseline)

Decision Tree: `{"validation_RMSE": 100.52561862530366, "validation_R2": 0.2822453150965425}`

Random Forest: `{"validation_RMSE": 98.03204464049095, "validation_R2": 0.31741200068775266}`

Gradient Boosting: `{"validation_RMSE": 99.44261603457791, "validation_R2": 0.29762732318068996}`

I programmed my "find_best_model" function to pick the best one based on lowest RSME on my validation set, which turned out to be the Random Forest with these hyperparameters:

`{'criterion': 'squared_error', 'max_depth': 50, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 100}`

Given more time, I would've like to have tried more regression models, and possibly with a wider grid search. I would've liked to have narrowed down my grid search more, too.

## Milestone 3

In this milestone I tried applying 4 different classification models to my numerical data, with the label being the Category of the property (of which there were 5). The k-fold validation strategy inbuilt in GridsearchCV, as well as my final scores being based on my own separated validation set, prevented me from overfitting these models.

```python
# %%
import collections
import itertools
import joblib
import json
import numpy as np
import os
import pandas as pd

import tabular_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Product features and labels directly from the csv
clean_data = pd.read_csv("clean_tabular_data.csv")
features = clean_data.select_dtypes(include = ["int64", "float64"])
label_series = clean_data["Category"]

np.random.seed(2)

# Encode labels

label_categories = label_series.unique()
le = LabelEncoder()
label_encoded = le.fit_transform(label_series)

# Count how many of each label there is
label_count = collections.Counter(label_encoded)
for key, value in label_count.items():
   print(f"{key}: {value}")

X, y = (features, label_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.5
)

# %%

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
    np.random.seed(2)
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
    best_model_list = [best_model.fit(X_train, y_train), GS.best_params_, performance_metrics_dict]
    print(best_model_list)
    return best_model_list

def save_model(model_list, folder="models/classification/logistic_regression"):
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

def evaluate_all_models(task_folder="models/classification"):
    np.random.seed(2)
    logistic_regression_model = tune_classification_model_hyperparameters("LogisticRegression", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "tol": [1E-5, 1E-4, 1E-3],
    "max_iter": [100, 500, 1000],
    "multi_class": ["multinomial"]
    })

    save_model(logistic_regression_model, folder=f"{task_folder}/logistic_regression")
    
    np.random.seed(2)
    decision_tree_model = tune_classification_model_hyperparameters("DecisionTreeClassifier", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "criterion": ["gini", "entropy"],
    "max_depth": [15, 30, 45, 60],
    "min_samples_split": [2, 4, 0.2, 0.4],
    "max_features": [4, 6, 8]
    })

    save_model(decision_tree_model, folder=f"{task_folder}/decision_tree")

    np.random.seed(2)
    random_forest_model = tune_classification_model_hyperparameters("RandomForestClassifier", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [50, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [30, 40, 50],
    "min_samples_split": [2, 0.1, 0.2],
    "max_features": [1, 2, 3]
    })

    save_model(random_forest_model, folder=f"{task_folder}/random_forest")

    np.random.seed(2)
    gradient_boosting_model = tune_classification_model_hyperparameters("GradientBoostingClassifier", 
    X_train, y_train, X_validation, y_validation, search_space = 
    {
    "n_estimators": [25, 50, 100],
    "loss": ["log_loss"],
    "max_depth": [1, 3, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_features": [1, 2, 3]
    })

    save_model(gradient_boosting_model, folder=f"{task_folder}/gradient_boosting")

    return logistic_regression_model, decision_tree_model, random_forest_model, gradient_boosting_model

def find_best_model(model_details_list):
    validation_scores = [x[2]["validation_accuracy"] for x in model_details_list]
    best_score_index = np.argmax(validation_scores)
    best_model_details = model_details_list[best_score_index]
    return best_model_details

if  __name__ == '__main__':
    y_hat = create_first_model()
    # print_performance(y_hat[0], y_hat[1])
    model_details_list = evaluate_all_models()
    best_model_details = find_best_model(model_details_list)
    print(f"The best model: {best_model_details}")
```

The metrics of the 4 models I created:

Logistic Regression: `{"validation_accuracy": 0.36, "validation_f1": 0.35727000877587234}`

Decision Tree: `{"validation_accuracy": 0.4, "validation_f1": 0.39440586327045535}`

Random Forest: `{"validation_accuracy": 0.384, "validation_f1": 0.33516113516113516}`

Gradient Boosting: `{"validation_accuracy": 0.472, "validation_f1": 0.4681900452488687}`

I programmed my "find_best_model" function to pick the best one based on highest accuracy on my validation set, which turned out to be the Gradient Boosted Classifier with these hyperparameters:

`{"learning_rate": 0.05, "loss": "log_loss", "max_depth": 3, "max_features": 1, "n_estimators": 100}`

Given more time, I would've like to have tried some more different classifier models, and possibly with a wider grid search. I would've liked to have narrowed down my grid search more, too.