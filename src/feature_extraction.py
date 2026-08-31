import numpy as np

def get_frequency_features(window, fs=500):
    fft_values = np.fft.rfft(window)
    fft_magnitude = np.abs(fft_values)

    freqs = np.fft.rfftfreq(len(window), d=1/fs)

    low_band = (freqs >= 0) & (freqs < 0.5)
    low_freq_energy = np.sum(fft_magnitude[low_band])

    high_band = (freqs >= 20) & (freqs <= 25)
    high_freq_energy = np.sum(fft_magnitude[high_band])

    total_energy = np.sum(fft_magnitude) + 1e-8

    return {
        "low_freq_ratio": low_freq_energy / total_energy,
        "high_freq_ratio": high_freq_energy / total_energy
    }

def extract_features(window):
    features = {}

    features["mean"] = np.mean(window)
    features["std"] = np.std(window)
    features["variance"] = np.var(window)
    features["rms"] = np.sqrt(np.mean(window**2))
    features["min"] = np.min(window)
    features["max"] = np.max(window)
    features["range"] = np.max(window) - np.min(window)

    return features


from segment_data import all_windows
import pandas as pd

feature_rows = []

for entry in all_windows:
    window = entry["window"]
    label = entry["label"]

    features = extract_features(window)
    freq_features = get_frequency_features(window)
    features.update(freq_features)
    features["label"] = label

    feature_rows.append(features)

feature_table = pd.DataFrame(feature_rows)

print(feature_table.shape)
print(feature_table.head())
feature_table.to_csv("results/feature_table.csv", index=False)
print("Feature table saved to results/feature_table.csv")