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
# print(dataset[1])
# print(type(dataset))
# print(type(dataset[3][0]))
# print(type(dataset[3][1]))

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

hyper_dict = get_nn_config()

class NN(torch.nn.Module):
    def __init__(self, config=hyper_dict):
        super().__init__()
        # Define layers
        width = config["hidden_layer_width"]
        print(f"width: {width}")
        depth = config["depth"]
        print(f"depth: {depth}")
        layers = []
        layers.append(torch.nn.Linear(11, width))
        for hidden_layer in range(depth - 1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        print(f"Layers: {layers}")
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        # Use the layers to process the features
        processed_features = self.layers(X)
        return processed_features

def train(model, data_loader, epochs):
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
            print("Loss:", loss.item())
            optimizer.step() # optimisation step
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

    print(train_rmse_loss)
    print(train_r2_score)

    X_validation = torch.stack([tuple[0] for tuple in validation_set]).type(torch.float32)
    y_validation = torch.stack([torch.tensor(tuple[1]) for tuple in validation_set])
    y_validation = torch.unsqueeze(y_validation, 1)
    y_hat_validation = model(X_validation)
    validation_rmse_loss = torch.sqrt(F.mse_loss(y_hat_validation, y_validation.float()))
    validation_r2_score = 1 - validation_rmse_loss / torch.var(y_validation.float())

    print(validation_rmse_loss)
    print(validation_r2_score)

    X_test = torch.stack([tuple[0] for tuple in test_set]).type(torch.float32)
    y_test = torch.stack([torch.tensor(tuple[1]) for tuple in test_set])
    y_test = torch.unsqueeze(y_test, 1)
    y_hat_test = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_hat_test, y_test.float()))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test.float())

    print(test_rmse_loss)
    print(test_r2_score)

    RMSE_loss = [train_rmse_loss, validation_rmse_loss, test_rmse_loss]
    R_squared = [train_r2_score, validation_r2_score, test_r2_score]

    metrics_dict["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics_dict["R_squared"] = [score.item() for score in R_squared]

    return metrics_dict

def save_model(model, performance_metrics, nn_folder="models/regression/neural_networks"):
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
        hyper_params = get_nn_config()
        with open(f"{model_folder}/hyperparameters.json", 'w') as fp:
            json.dump(hyper_params, fp)
        # Save performance metrics
        with open(f"{model_folder}/metrics.json", 'w') as fp:
            json.dump(performance_metrics, fp)

def do_full_model_train(model, epochs=10):
    start_time = time.time()
    train(model, train_loader, epochs)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"It took {training_duration} seconds to train the model")
    metrics_dict = evaluate_model(model, training_duration, epochs)
    save_model(model, metrics_dict)

if  __name__ == '__main__':
    np.random.seed(2)
    model = NN()
    do_full_model_train(model, 2)
    sd = model.state_dict()
    print(sd)