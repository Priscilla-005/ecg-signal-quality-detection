import pandas as pd

feature_table = pd.read_csv("results/feature_table.csv")

summary = feature_table.groupby("label")[["std", "rms", "low_freq_ratio", "high_freq_ratio"]].mean()
summary.index = ["Standing", "Walking", "Jumping"]

print(summary)

summary.to_csv("results/class_averages.csv")
print("\nSaved to results/class_averages.csv")