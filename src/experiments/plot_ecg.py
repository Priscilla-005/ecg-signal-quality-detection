import wfdb
import matplotlib.pyplot as plt

# Load ECG record
record = wfdb.rdrecord(
    '100',
    pn_dir='mitdb',
    sampto=1000
)

# Get first ECG channel (MLII)
ecg_signal = record.p_signal[:, 0]

# Create time axis
time = [i / record.fs for i in range(len(ecg_signal))]

# Plot ECG
plt.figure(figsize=(12, 4))
plt.plot(time, ecg_signal)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal - Record 100")
plt.grid()

plt.show()
print("no.of.samples: ",len(ecg_signal))
print("maximim: ", ecg_signal.max())
print("minimum : ", ecg_signal.min())
print("mean : ", ecg_signal.mean())
print("standard deviation: ", ecg_signal.std())