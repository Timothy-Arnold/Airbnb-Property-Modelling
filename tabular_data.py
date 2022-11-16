import numpy as np
import pandas as pd
from ast import literal_eval

# print(raw_data)
# print(raw_data.info())
# print(raw_data.describe())
# print(raw_data["Value_rating"])
# print(raw_data[raw_data["Value_rating"].notna()])

def remove_rows_with_missing_ratings(raw_data):
    raw_data.dropna(subset=["Value_rating"], inplace=True)
    return raw_data

def combine_description_strings(desc):
    desc_list = literal_eval(desc.strip())[1:]
    desc_list_without_empty_quotes = list(filter(lambda x: x != "", desc_list))
    full_desc = ' '.join(desc_list_without_empty_quotes)
    return full_desc

def fix_description_strings(df):
    df.dropna(subset=["Description"], inplace=True)
    # Information for 586 was found to be completely in the wrong places
    df.drop([586], inplace=True)
    df["Description"] = df["Description"].apply(combine_description_strings)
    print(df.info())
    return df

def set_default_feature_values(df):
    df[["guests", "beds", "bathrooms", "bedrooms"]] = df[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1)
    return df

def clean_tabular_data(df):
    raw_data_with_ratings = remove_rows_with_missing_ratings(raw_data)
    raw_data_with_description = fix_description_strings(raw_data_with_ratings)
    raw_data_default_features = set_default_feature_values(raw_data_with_description)
    print(raw_data_default_features)
    print(raw_data_default_features.info())
    return raw_data_default_features

# desc = raw_data.iloc[10]["Description"]
# print(desc)
# print(type(desc))
# print("--------------------------------")
# desc_list = literal_eval(desc)[1:]
# print(desc_list)
# print(type(desc_list))
# desc_list_without_empty_quotes = list(filter(lambda x: x != "", desc_list))
# print(desc_list_without_empty_quotes)
# print("--------------------------------")
# new = ' '.join(desc_list_without_empty_quotes)
# print(new)

if __name__ == '__main__':
    raw_data = pd.read_csv("listing.csv")
    clean_data = clean_tabular_data(raw_data)
    clean_data.to_csv("clean_tabular_data.csv")