
with open("dataset/RECORDS", "r") as f:
    record_names = f.read().splitlines()
print("Total records found:", len(record_names))
print("First 5 record names:", record_names[:5])
def get_label_from_name(name):
    activity_letter = name[-1]
    label_map = {"s": 0, "w": 1, "j": 2}
    return label_map[activity_letter]

for name in record_names:
    label = get_label_from_name(name)
    print(name, "->", label)

import wfdb

first_record_name = record_names[0]
record = wfdb.rdrecord(f"dataset/{first_record_name}")

print("\nTesting signal load for:", first_record_name)
print("Sampling frequency:", record.fs)
print("Number of channels:", record.n_sig)
print("Signal shape:", record.p_signal.shape)

all_records = []

for name in record_names:
    record = wfdb.rdrecord(f"dataset/{name}")
    label = get_label_from_name(name)
    all_records.append({
        "name": name,
        "signal": record.p_signal,
        "fs": record.fs,
        "label": label
    })

print("\nTotal records loaded:", len(all_records))
print("Example entry keys:", all_records[0].keys())
print("Example entry - name:", all_records[0]["name"])
print("Example entry - label:", all_records[0]["label"])
print("Example entry - signal shape:", all_records[0]["signal"].shape)