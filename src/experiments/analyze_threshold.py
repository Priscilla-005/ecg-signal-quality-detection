import pandas as pd

feature_table = pd.read_csv("results/feature_table.csv")

# Group by actual label and look at the real distribution of std values
print(feature_table.groupby("label")["std"].describe())
print("\n")
print(feature_table.groupby("label")["high_freq_ratio"].describe())