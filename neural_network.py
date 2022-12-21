import numpy as np
import pandas as pd
import random
import tabular_data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

transform = transforms.PILToTensor()

clean_data = pd.read_csv("clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
features = X.to_numpy()
label = y.to_numpy()

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
print(dataset[3])
print(type(dataset))
print(type(dataset[3][0]))
print(len(dataset))

train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.85), len(dataset) - int(len(dataset) * 0.85)])
train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set) * 14/17), len(train_set) - int(len(train_set) * 14/17)])

print(len(train_set))
print(len(validation_set))
print(len(test_set))

train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=len(validation_set), shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)

print(type(train_set))
print(type(train_loader))

# for batch in test_loader:
#     print(batch)
#     features, labels = batch
#     print(features.shape)
#     print(labels.shape)
#     break