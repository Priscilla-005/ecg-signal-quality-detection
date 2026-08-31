import streamlit as st
import wfdb
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("results/final_rf_model.pkl")

st.title("ECG Signal Quality & Artifact Detection")
st.write("Select an ECG record to check whether it reflects standing, walking, or jumping activity.")

# Load the list of available records
with open("dataset/RECORDS", "r") as f:
    record_names = f.read().splitlines()

selected_record = st.selectbox("Choose a record", record_names)

if st.button("Analyze"):
    # Load the selected record
    record = wfdb.rdrecord(f"dataset/{selected_record}")
    signal = record.p_signal[:, 0]

    # Take the first 1-second window (500 samples) to match our trained model
    window = signal[:500]

    # --- Statistical features ---
    mean = np.mean(window)
    std = np.std(window)
    variance = np.var(window)
    rms = np.sqrt(np.mean(window**2))
    min_val = np.min(window)
    max_val = np.max(window)
    range_val = max_val - min_val

    # --- Frequency features ---
    fft_values = np.fft.rfft(window)
    fft_magnitude = np.abs(fft_values)
    freqs = np.fft.rfftfreq(len(window), d=1/500)

    low_band = (freqs >= 0) & (freqs < 0.5)
    low_freq_energy = np.sum(fft_magnitude[low_band])

    high_band = (freqs >= 20) & (freqs <= 25)
    high_freq_energy = np.sum(fft_magnitude[high_band])

    total_energy = np.sum(fft_magnitude) + 1e-8
    low_freq_ratio = low_freq_energy / total_energy
    high_freq_ratio = high_freq_energy / total_energy

    # --- Combine into the same column order used during training ---
    features = [[mean, std, variance, rms, min_val, max_val, range_val, low_freq_ratio, high_freq_ratio]]

    # --- Model prediction ---
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]

    activity_names = {0: "Standing", 1: "Walking", 2: "Jumping"}
    artifact_expectation = {
        0: "Minimal motion artifact expected - subject stationary",
        1: "Moderate, rhythmic motion artifact expected - continuous movement",
        2: "Severe, transient motion artifact expected - sudden impact"
    }

    predicted_activity = activity_names[prediction]
    confidence = prediction_proba[prediction] * 100

    st.subheader(f"Predicted Activity: {predicted_activity}")
    st.write(f"Artifact Expectation: {artifact_expectation[prediction]}")
    st.write(f"Model confidence: {confidence:.1f}%")

    st.write("**Confidence breakdown:**")
    st.json({
        "Standing": f"{prediction_proba[0]*100:.1f}%",
        "Walking": f"{prediction_proba[1]*100:.1f}%",
        "Jumping": f"{prediction_proba[2]*100:.1f}%"
    })

    # --- Plot the signal ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(window)
    ax.set_title(f"ECG Signal - {selected_record} (first 1 second)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (mV)")
    st.pyplot(fig)