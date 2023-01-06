import joblib
import json
import numpy as np
import os
import pandas as pd
import random
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

print("Size of train set: " + str(len(train_set)))
print("Size of validation set: " + str(len(validation_set)))
print("Size of test set: " + str(len(test_set)))

batch_size = 8

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(f"The type of the train set: {type(train_set)}")

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
            print(loss.item())
            optimizer.step() # optimisation step
            optimizer.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_idx)
            batch_idx += 1

def evaluate_model(model, training_duration):
    metrics_dict = {"training_duration": training_duration}
    return metrics_dict

def save_model(model, performance_metrics, nn_folder="models/regression/neural_networks"):
    if not isinstance(model, torch.nn.Module):
        print("Error: Model is not a Pytorch Module!")
    else:
        # Make model folder
        if not os.path.exists(nn_folder):
            os.makedirs(nn_folder)
        print("Made nn folder")
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
    metrics_dict = evaluate_model(model, training_duration)
    save_model(model, metrics_dict)

if  __name__ == '__main__':
    np.random.seed(2)
    model = NN()
    do_full_model_train(model)
    sd = model.state_dict()
    print(sd)