def segment_signal(signal, window_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows
from load_dataset import all_records

WINDOW_SIZE = 500 

all_windows = []  

for entry in all_records:
    signal = entry["signal"][:, 0] 
    label = entry["label"]

    windows = segment_signal(signal, WINDOW_SIZE)

    for w in windows:
        all_windows.append({
            "window": w,
            "label": label,
            "source_record": entry["name"]
        })

print("Total windows created:", len(all_windows))
print("One window's shape:", all_windows[0]["window"].shape)
print("One window's label:", all_windows[0]["label"])