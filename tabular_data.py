import numpy as np
import pandas as pd

print("Start Reading!")
raw_data = pd.read_csv("listing.csv")
# print(raw_data)
# print(raw_data.info())
# print(raw_data.describe())
# print(raw_data["Value_rating"])
# print(raw_data[raw_data["Value_rating"].notna()])

def remove_rows_with_missing_ratings(raw_data):
    raw_data.dropna(subset=["Value_rating"], inplace=True)
    return raw_data

raw_data_with_ratings = remove_rows_with_missing_ratings(raw_data)
print(raw_data_with_ratings)