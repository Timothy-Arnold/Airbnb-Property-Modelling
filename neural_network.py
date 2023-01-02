import numpy as np
import pandas as pd
import random
import tabular_data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

np.random.seed(2)

transform = transforms.PILToTensor()

clean_data = pd.read_csv("clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
print(type(X))
features = X.to_numpy()
print(type(features))
label = y.to_numpy()
number_of_features = len(features[1])
print(f"Number of features: {number_of_features}")

np.random.seed(2)

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = features
        self.y = label

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)

dataset = AirbnbNightlyPriceImageDataset()
# print(dataset[3])
# print(type(dataset))
# print(type(dataset[3][0]))

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 17/20), len(dataset) - int(len(dataset) * 17/20)])
train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 14/17), len(train_set) - int(len(train_set) * 14/17)])

print("Size of train set: " + str(len(train_set)))
print("Size of validation set: " + str(len(validation_set)))
print("Size of test set: " + str(len(test_set)))

batch_size = 4

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(f"The type of the train set: {type(train_set)}")
# print(type(train_loader))
example = next(iter(train_loader))
# print(example)
features, label = example
features = features.reshape(batch_size, -1)

class NeuralNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Initialize parameters
        self.linear_layer = torch.nn.Linear(number_of_features, 1)

    def forward(self, features):
        # Use the layers to process the features
        features = features.type(torch.float)
        return self.linear_layer(features)

def train(model, data_loader, epochs):
    for epoch in range(epochs):
        example = next(iter(data_loader))
        features, label = example
        print(features.size())
        features = features.reshape(batch_size, -1)
        print(model(features))

if  __name__ == '__main__':
    np.random.seed(2)
    train(NeuralNetwork(), train_loader, 1)