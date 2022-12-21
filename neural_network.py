import pandas as pd
import tabular_data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


clean_data = pd.read_csv("clean_tabular_data.csv")
X, y = tabular_data.load_airbnb(clean_data, "Price_Night")
print(type(X))
print(type(y))

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = (X, y)

    def __getitem__(self, index):
        features = self.data[0].iloc[index]
        features_tensor = torch.tensor(features)
        label = self.data[1].iloc[index]
        return (features_tensor, label)

    def __len__(self):
        return len(self.data[0])

dataset = AirbnbNightlyPriceImageDataset()
print(dataset[3])
print(type(dataset[3][0]))
print(len(dataset))

# train_loader = DataLoader(dataset, batch_size=)