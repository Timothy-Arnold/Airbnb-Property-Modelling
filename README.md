# Airbnb-Property-Modelling

In this project I built frameworks to systematically train, tune, and evaluate models on several tasks tackled by the Airbnb team.
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

hyper_param_dict = {
    "DecisionTreeRegressor": {
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [15, 30, 45, 60],
        "min_samples_split": [2, 4, 0.2, 0.4],
        "max_features": [4, 6, 8]
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 150],
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": [30, 40, 50],
        "min_samples_split": [2, 0.1, 0.2],
        "max_features": [1, 2]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [25, 50, 100],
        "loss": ["squared_error", "absolute_error"],
        "max_depth": [1, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_features": [1, 2, 3]
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
    best_model_details = min(fitted_model_list, key=lambda x: x[2]["validation_RMSE"])
    print(f"Best model by validation_RMSE metric:\n{best_model_details}")
    return best_model_details

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

def save_model(model_details, folder="models/regression_price/linear_regression"):
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

def evaluate_all_models(task_folder="models/regression_price"):
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
    X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5
    )
    model_details_dict = evaluate_all_models()
    best_model_details = find_best_model(model_details_dict)
    print(f"The best model: {best_model_details}")
```

The metrics of the 4 models I created:

**Linear Regression:** `{"validation_RMSE": 2903929721.9552526, "validation_R2": -598955739200091.6}` (was just to form a baseline)

**Decision Tree:** `{"validation_RMSE": 100.52561862530366, "validation_R2": 0.2822453150965425}`

**Random Forest:** `{"validation_RMSE": 98.03204464049095, "validation_R2": 0.31741200068775266}`

**Gradient Boosting:** `{"validation_RMSE": 99.44261603457791, "validation_R2": 0.29762732318068996}`

I programmed my "find_best_model" function to pick the best one based on lowest RSME on my validation set, which turned out to be the Random Forest with these hyperparameters:

`{'criterion': 'squared_error', 'max_depth': 50, 'max_features': 2, 'min_samples_split': 2, 'n_estimators': 100}`

Given more time, I would've like to have tried more regression models, and possibly with a wider grid search. I would've liked to have narrowed down my grid search more, too.

## Milestone 3

In this milestone I tried applying 4 different classification models to my numerical data, with the label being the Category of the property (of which there were 5). The k-fold validation strategy inbuilt in GridsearchCV, as well as my final scores being based on my own separated validation set, prevented me from overfitting these models.

```python

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
```

The metrics of the 4 models I created:

**Logistic Regression:** `{"validation_accuracy": 0.36, "validation_f1": 0.35727000877587234}`

**Decision Tree:** `{"validation_accuracy": 0.4, "validation_f1": 0.39440586327045535}`

**Random Forest:** `{"validation_accuracy": 0.384, "validation_f1": 0.33516113516113516}`

**Gradient Boosting:** `{"validation_accuracy": 0.472, "validation_f1": 0.4681900452488687}`

I programmed my "find_best_model" function to pick the best one based on highest accuracy on my validation set, which turned out to be the Gradient Boosted Classifier with these hyperparameters:

`{"learning_rate": 0.05, "loss": "log_loss", "max_depth": 3, "max_features": 1, "n_estimators": 100}`

Given more time, I would've like to have tried some more different classifier models, and possibly with a wider grid search. I would've liked to have narrowed down my grid search more, too.

## Milestone 4

In this Milestone I used Pytorch to make a neural network to predict my numerical data, hopefully to a much better degree of accuracy than my previous regression models. I started off by creating my own Pytorch Dataset, splitting the data into train/validation/test, then making dataloaders that would split these sets up into batches.

```python
import itertools
import json
import numpy as np
import os
import pandas as pd
import tabular_data
import torch
import torch.nn.functional as F
import yaml

import time
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split 
from torch.utils.tensorboard import SummaryWriter

np.random.seed(2)

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv("clean_tabular_data.csv")
        self.features, self.label = tabular_data.load_airbnb(clean_data, "Price_Night")

    def __getitem__(self, index):
        features = self.features.iloc[index]
        features = torch.tensor(features)
        label = self.label.iloc[index]
        return (features, label)

    def __len__(self):
        return len(self.features)

dataset = AirbnbNightlyPriceImageDataset()

train_set, test_set = random_split(dataset, [int(len(dataset) * 17/20), len(dataset) - int(len(dataset) * 17/20)])
train_set, validation_set = random_split(train_set, [int(len(train_set) * 14/17), len(train_set) - int(len(train_set) * 14/17)])
print(f"The type of the train set: {type(train_set)}")
print("Size of train set: " + str(len(train_set)))
print("Size of validation set: " + str(len(validation_set)))
print("Size of test set: " + str(len(test_set)))

batch_size = 8

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
```

I then defined the NN using an initial config just for starters.

```python
def get_nn_config():
    with open("nn_config.yaml", 'r') as stream:
        try:
            hyper_dict = yaml.safe_load(stream)
            print(hyper_dict)
        except yaml.YAMLError as error:
            print(error)
    return hyper_dict

# hyper_dict_example = get_nn_config()

class NN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
        width = config["hidden_layer_width"]
        depth = config["depth"]
        layers = []
        layers.append(torch.nn.Linear(11, width))
        for hidden_layer in range(depth - 1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        # Use the layers to process the features
        processed_features = self.layers(X)
        return processed_features
```

I then created functions to train the NN and evaluate the model it made.

```python
def train(model, data_loader, hyper_dict, epochs):
    optimizer_class = hyper_dict["optimizer"]
    optimizer_instance = getattr(torch.optim, optimizer_class)
    optimizer = optimizer_instance(model.parameters(), lr=hyper_dict["learning_rate"])

    writer = SummaryWriter()

    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            features = features.type(torch.float32)
            # Make labels the same shape as predictions
            labels = torch.unsqueeze(labels, 1)
            prediction = model(features)
            loss = F.mse_loss(prediction, labels.float())
            loss.backward()
            print("Loss:", loss.item())
            # Optimisation step
            optimizer.step() 
            optimizer.zero_grad()
            # Add loss to Tensorboard graph
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1

def evaluate_model(model, training_duration, epochs):
    # Initialize performance metrics dictionary
    metrics_dict = {"training_duration": training_duration}

    number_of_predictions = epochs * len(train_set)
    inference_latency = training_duration / number_of_predictions
    metrics_dict["inference_latency"] = inference_latency

    X_train = torch.stack([tuple[0] for tuple in train_set]).type(torch.float32)
    y_train = torch.stack([torch.tensor(tuple[1]) for tuple in train_set])
    y_train = torch.unsqueeze(y_train, 1)
    y_hat_train = model(X_train)
    train_rmse_loss = torch.sqrt(F.mse_loss(y_hat_train, y_train.float()))
    train_r2_score = 1 - train_rmse_loss / torch.var(y_train.float())

    print("Train RMSE:", train_rmse_loss.item())
    print("Train R2:", train_r2_score.item())

    X_validation = torch.stack([tuple[0] for tuple in validation_set]).type(torch.float32)
    y_validation = torch.stack([torch.tensor(tuple[1]) for tuple in validation_set])
    y_validation = torch.unsqueeze(y_validation, 1)
    y_hat_validation = model(X_validation)
    validation_rmse_loss = torch.sqrt(F.mse_loss(y_hat_validation, y_validation.float()))
    validation_r2_score = 1 - validation_rmse_loss / torch.var(y_validation.float())

    print("Validation RMSE:", validation_rmse_loss.item())
    print("Validation R2:", validation_r2_score.item())

    X_test = torch.stack([tuple[0] for tuple in test_set]).type(torch.float32)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_set])
    y_test = torch.unsqueeze(y_test, 1)
    y_hat_test = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_hat_test, y_test.float()))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test.float())

    print("Test RMSE:", test_rmse_loss.item())
    print("Test R2:", test_r2_score.item())

    RMSE_loss = [train_rmse_loss, validation_rmse_loss, test_rmse_loss]
    R_squared = [train_r2_score, validation_r2_score, test_r2_score]

    metrics_dict["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics_dict["R_squared"] = [score.item() for score in R_squared]

    return metrics_dict
```

I made a function to save the model to my local directory, under folders named after their times of training.

```python
def save_model(model, hyper_dict, performance_metrics, nn_folder="models/regression_price/neural_networks"):
    if not isinstance(model, torch.nn.Module):
        print("Error: Model is not a Pytorch Module!")
    else:
        # Make model folder
        if not os.path.exists(nn_folder):
            os.makedirs(nn_folder)
        # Name folder as current time
        save_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        model_folder = nn_folder + "/" + save_time
        os.mkdir(model_folder)
        # Save model
        torch.save(model.state_dict(), f"{model_folder}/model.pt")
        # Save hyper parameters
        with open(f"{model_folder}/hyperparameters.json", 'w') as fp:
            json.dump(hyper_dict, fp)
        # Save performance metrics
        with open(f"{model_folder}/metrics.json", 'w') as fp:
            json.dump(performance_metrics, fp)
```

I then brought all these functions together to train models with particular hyperparameters

```python
def do_full_model_train(hyper_dict, epochs=5):
    model = NN(hyper_dict)
    start_time = time.time()
    train(model, train_loader, hyper_dict, epochs)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"It took {training_duration} seconds to train the model")
    metrics_dict = evaluate_model(model, training_duration, epochs)
    save_model(model, hyper_dict, metrics_dict)
    print(hyper_dict)
    model_info = [model, hyper_dict, metrics_dict]
    return model_info
```

And then made functions to do a gridsearch of NN's, picking the one with the lowest RMSE_loss on the validation set.

```python
def generate_nn_configs():
    hyper_values_dict_list = []
    search_space = {
    'optimizer': ['SGD', "Adam"],
    'learning_rate': [0.0001, 0.001],
    'hidden_layer_width': [5, 10],
    'depth': [2, 3]
    }
    keys = search_space.keys()
    vals = search_space.values()
    # Find all combindations of hyperparameters
    for instance in itertools.product(*vals):
        hyper_values_dict = dict(zip(keys, instance))
        hyper_values_dict_list.append(hyper_values_dict)

    return hyper_values_dict_list

def find_best_nn(epochs=10):
    lowest_RMSE_loss_validation = np.inf
    hyper_values_dict_list = generate_nn_configs()
    for hyper_values_dict in hyper_values_dict_list:
        model_info = do_full_model_train(hyper_values_dict, epochs)
        metrics_dict = model_info[2]
        RMSE_loss = metrics_dict["RMSE_loss"]
        RMSE_loss_validation = RMSE_loss[1]
        print(hyper_values_dict)
        print(RMSE_loss_validation)
        print("-" * 80)
        if RMSE_loss_validation < lowest_RMSE_loss_validation:
            lowest_RMSE_loss_validation = RMSE_loss_validation
            best_model_info = model_info
        # Pause to make sure NNs are saved under folders with different names
        time.sleep(1)

    best_model, best_hyper_dict, best_metrics_dict = best_model_info
    print("Best Model:", "\n", best_hyper_dict, best_metrics_dict)

    save_model(best_model, best_hyper_dict, best_metrics_dict, "models/regression_price/neural_networks/best_neural_networks")

if  __name__ == '__main__':
    find_best_nn(20)
```

I saved the best model from each Gridsearch in their own folder in my local directory. I chose to use 20 epochs because using 30 tended to lead to overfitting, while having less than 15 epochs tended to lead to underfitting. I also found that learning rates above 0.001 lead to explosions in loss, so I didn't try any higher ones in my gridsearch.

The loss curve for each mini-batch of a sample of 16 models within my gridsearch, with smoothing in Tensorboard:

![loss_all](screenshots/loss_all.jpg)

The loss curve for the best model:

![loss_best](screenshots/loss_best.jpg)

The shape of my best performing neural network:

![neural_network](screenshots/neural_network.jpg)

## Milestone 5

For my final milestone I changed the dataset to have "bedrooms" as the label, with "price_night" as one of the features. I also added "category" as a feature using one-hot encoding (since it had only 5 categories) by using `pd.get_dummies` and concatenating the dataframes.

New neural network code:

```python
import itertools
import json
import numpy as np
import os
import pandas as pd
import tabular_data
import torch
import torch.nn.functional as F
import yaml

import time
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split 
from torch.utils.tensorboard import SummaryWriter

np.random.seed(2)

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv("clean_tabular_data.csv")
        numerical_data = tabular_data.load_airbnb(clean_data, "bedrooms")
        category_series = clean_data["Category"]
        category_options = category_series.unique()
        one_hot = pd.get_dummies(category_series)
        print(category_options)
        one_hot = one_hot.astype("int64")
        numerical_data[0] = pd.concat([numerical_data[0], one_hot], axis=1)
        self.features, self.label = numerical_data

    def __getitem__(self, index):
        features = self.features.iloc[index]
        features = torch.tensor(features)
        label = self.label.iloc[index]
        return (features, label)

    def __len__(self):
        return len(self.features)

dataset = AirbnbNightlyPriceImageDataset()

train_set, test_set = random_split(dataset, [int(len(dataset) * 17/20), len(dataset) - int(len(dataset) * 17/20)])
train_set, validation_set = random_split(train_set, [int(len(train_set) * 14/17), len(train_set) - int(len(train_set) * 14/17)])
print(f"The type of the train set: {type(train_set)}")
print("Size of train set: " + str(len(train_set)))
print("Size of validation set: " + str(len(validation_set)))
print("Size of test set: " + str(len(test_set)))

batch_size = 8

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

def get_nn_config():
    with open("nn_config.yaml", 'r') as stream:
        try:
            hyper_dict = yaml.safe_load(stream)
            print(hyper_dict)
        except yaml.YAMLError as error:
            print(error)
    return hyper_dict

# hyper_dict_example = get_nn_config()

class NN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
        width = config["hidden_layer_width"]
        depth = config["depth"]
        layers = []
        layers.append(torch.nn.Linear(16, width))
        for hidden_layer in range(depth - 1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        # print(f"Layers: {layers}")
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        # Use the layers to process the features
        processed_features = self.layers(X)
        return processed_features

def train(model, data_loader, hyper_dict, epochs):
    optimizer_class = hyper_dict["optimizer"]
    optimizer_instance = getattr(torch.optim, optimizer_class)
    optimizer = optimizer_instance(model.parameters(), lr=hyper_dict["learning_rate"])

    writer = SummaryWriter()

    batch_idx = 0

    for epoch in range(epochs):
        for batch in data_loader:
            features, labels = batch
            features = features.type(torch.float32)
            labels = torch.unsqueeze(labels, 1)
            prediction = model(features)
            # Make labels the same shape as predictions
            loss = F.mse_loss(prediction, labels.float())
            loss.backward()
            # print("Loss:", loss.item())
            # Optimisation step
            optimizer.step() 
            optimizer.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1

def evaluate_model(model, training_duration, epochs):
    # Initialize performance metrics dictionary
    metrics_dict = {"training_duration": training_duration}

    number_of_predictions = epochs * len(train_set)
    inference_latency = training_duration / number_of_predictions
    metrics_dict["inference_latency"] = inference_latency

    X_train = torch.stack([tuple[0] for tuple in train_set]).type(torch.float32)
    y_train = torch.stack([torch.tensor(tuple[1]) for tuple in train_set])
    y_train = torch.unsqueeze(y_train, 1)
    y_hat_train = model(X_train)
    train_rmse_loss = torch.sqrt(F.mse_loss(y_hat_train, y_train.float()))
    train_r2_score = 1 - train_rmse_loss / torch.var(y_train.float())

    print("Train RMSE:", train_rmse_loss.item())
    print("Train R2:", train_r2_score.item())

    X_validation = torch.stack([tuple[0] for tuple in validation_set]).type(torch.float32)
    y_validation = torch.stack([torch.tensor(tuple[1]) for tuple in validation_set])
    y_validation = torch.unsqueeze(y_validation, 1)
    y_hat_validation = model(X_validation)
    validation_rmse_loss = torch.sqrt(F.mse_loss(y_hat_validation, y_validation.float()))
    validation_r2_score = 1 - validation_rmse_loss / torch.var(y_validation.float())

    print("Validation RMSE:", validation_rmse_loss.item())
    print("Validation R2:", validation_r2_score.item())

    X_test = torch.stack([tuple[0] for tuple in test_set]).type(torch.float32)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_set])
    y_test = torch.unsqueeze(y_test, 1)
    y_hat_test = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_hat_test, y_test.float()))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test.float())

    print("Test RMSE:", test_rmse_loss.item())
    print("Test R2:", test_r2_score.item())

    RMSE_loss = [train_rmse_loss, validation_rmse_loss, test_rmse_loss]
    R_squared = [train_r2_score, validation_r2_score, test_r2_score]

    metrics_dict["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics_dict["R_squared"] = [score.item() for score in R_squared]

    return metrics_dict

def save_model(model, hyper_dict, performance_metrics, nn_folder="models/regression_bedrooms/neural_networks"):
    if not isinstance(model, torch.nn.Module):
        print("Error: Model is not a Pytorch Module!")
    else:
        # Make model folder
        if not os.path.exists(nn_folder):
            os.makedirs(nn_folder)
        save_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        model_folder = nn_folder + "/" + save_time
        os.mkdir(model_folder)
        # Save model
        torch.save(model.state_dict(), f"{model_folder}/model.pt")
        # Get and save hyper parameters
        with open(f"{model_folder}/hyperparameters.json", 'w') as fp:
            json.dump(hyper_dict, fp)
        # Save performance metrics
        with open(f"{model_folder}/metrics.json", 'w') as fp:
            json.dump(performance_metrics, fp)

def do_full_model_train(hyper_dict, epochs=10):
    model = NN(hyper_dict)
    start_time = time.time()
    train(model, train_loader, hyper_dict, epochs)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"It took {training_duration} seconds to train the model")
    metrics_dict = evaluate_model(model, training_duration, epochs)
    save_model(model, hyper_dict, metrics_dict)
    model_info = [model, hyper_dict, metrics_dict]
    return model_info

def generate_nn_configs():
    hyper_values_dict_list = []
    search_space = {
    'optimizer': ['Adam', "AdamW"],
    'learning_rate': [0.001, 0.002],
    'hidden_layer_width': [10, 15, 20, 25],
    'depth': [6, 8, 10, 12]
    }
    keys = search_space.keys()
    vals = search_space.values()
    for instance in itertools.product(*vals):
        hyper_values_dict = dict(zip(keys, instance))
        hyper_values_dict_list.append(hyper_values_dict)

    print(hyper_values_dict_list)
    return hyper_values_dict_list

def find_best_nn(epochs=10):
    lowest_RMSE_loss_validation = np.inf
    hyper_values_dict_list = generate_nn_configs()
    for hyper_values_dict in hyper_values_dict_list:
        model_info = do_full_model_train(hyper_values_dict, epochs)
        metrics_dict = model_info[2]
        RMSE_loss = metrics_dict["RMSE_loss"]
        RMSE_loss_validation = RMSE_loss[1]
        print(hyper_values_dict)
        print(RMSE_loss_validation)
        print("-" * 80)
        if RMSE_loss_validation < lowest_RMSE_loss_validation:
            lowest_RMSE_loss_validation = RMSE_loss_validation
            best_model_info = model_info
        time.sleep(1)

    best_model, best_hyper_dict, best_metrics_dict = best_model_info
    print("Best Model:", "\n", best_hyper_dict, best_metrics_dict)

    save_model(best_model, best_hyper_dict, best_metrics_dict, "models/regression_bedrooms/neural_networks/best_neural_networks")

if  __name__ == '__main__':
    np.random.seed(2)
    find_best_nn(20)
```
Loss of neural network in one gridsearch (from Tensorboard):

![loss_all](screenshots/loss_all_beds.jpg)

New code for the other traditional models:
```python
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
```

Example of running this code to get the best traditional model:

![loss_all](screenshots/traditional_models_running.png)

The metrics of the 4 models I created:

**Neural Network:** `{"validation_RMSE": 0.37593674659729004, "validation_R2": 0.43938618898391724}`

**Decision Tree:** `{"validation_RMSE": 0.6995236474439827, "validation_R2": 0.3429156640311677}`

**Random Forest:** `{"validation_RMSE": 0.3247990147768309, "validation_R2": 0.8583404950154693}`

**Gradient Boosting:** `{"validation_RMSE": 0.35296892441190447, "validation_R2": 0.832702574982145}`

The best model turned out to be the Random Forest with these hyperparameters:

`{"criterion": "absolute_error", "max_depth": 50, "max_features": 6, "min_samples_split": 2, "n_estimators": 100}`

## Conclusion

I made and tuned several models, including neural networks, to predict 3 different labels: Price per Night, Category of the property, and Number of Bedrooms. 

The models which I got the best results with after applying grid search to them:

**Price per Night:** Neural Network

**Category of the property:** Gradient Boosting Classifier

**Number of Bedrooms:** Random Forest Regressor

Given more time I would have liked to have made use of my cleaned image data to help train a CNN.