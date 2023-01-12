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