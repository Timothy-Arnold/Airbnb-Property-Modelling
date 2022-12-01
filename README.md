# Airbnb-Property-Modelling

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