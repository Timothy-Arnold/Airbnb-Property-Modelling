import numpy as np
import pandas as pd
import random
import tabular_data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split 
import torch.nn.functional as F
from torchvision import transforms

np.random.seed(2)

transform = transforms.PILToTensor()

np.random.seed(2)

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        clean_data = pd.read_csv("clean_tabular_data.csv")
        self.features, self.label = tabular_data.load_airbnb(clean_data, "Price_Night")
        print(type(self.features))
        print(type(self.label))

    def __getitem__(self, index):
        features = self.features.iloc[index]
        features = torch.tensor(features)
        label = self.label.iloc[index]
        return (features, label)

    def __len__(self):
        return len(self.features)

dataset = AirbnbNightlyPriceImageDataset()
print(dataset[1])
print(type(dataset))
print(type(dataset[3][0]))
print(type(dataset[3][1]))

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 17/20), len(dataset) - int(len(dataset) * 17/20)])
train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 14/17), len(train_set) - int(len(train_set) * 14/17)])

print("Size of train set: " + str(len(train_set)))
print("Size of validation set: " + str(len(validation_set)))
print("Size of test set: " + str(len(test_set)))

batch_size = 8

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print(f"The type of the train set: {type(train_set)}")

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, X):
        # Use the layers to process the features
        return self.layers(X)

def train(model, data_loader, epochs):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

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
            optimiser.step() # optimisation step
            optimiser.zero_grad()

if  __name__ == '__main__':
    np.random.seed(2)
    model = NN()
    train(NN(), train_loader, 1)