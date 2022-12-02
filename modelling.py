import numpy as np
import pandas as pd
import tabular_data

clean_data = pd.read_csv("clean_tabular_data.csv")
full_data = tabular_data.load_airbnb(clean_data, "Price_Night")