import numpy as np
from scipy.signal import find_peaks

def detect_quality(window, fs=500):

    peak_height = np.mean(window) + 0.5 * np.std(window)

    min_gap_between_beats = int(0.3 * fs)

    peaks, _ = find_peaks(window, height=peak_height, distance=min_gap_between_beats)
    num_peaks_found = len(peaks)

    if num_peaks_found < 2:
        return {
            "quality_label": 1,  # 1 = Poor quality
            "num_peaks": num_peaks_found,
            "rr_std": None,
            "reason": "Couldn't find enough clear heartbeats to judge rhythm"
        }
    beat_gaps = np.diff(peaks)

    average_gap = np.mean(beat_gaps)
    gap_variation = np.std(beat_gaps)

    if average_gap > 0:
        irregularity_score = gap_variation / average_gap
    else:
        irregularity_score = 999  
    if irregularity_score < 0.15:
        quality_label = 0  # Good quality
    else:
        quality_label = 1  # Poor quality

    return {
        "quality_label": quality_label,
        "num_peaks": num_peaks_found,
        "rr_std": gap_variation,
        "irregularity": irregularity_score,
        "reason": f"Heartbeat spacing irregularity score: {irregularity_score:.3f}"
    }

if __name__ == "__main__":
    from segment_data import all_windows

    # Peek at a few different windows - roughly one from each activity type
    print("Testing quality detection on a few sample windows:\n")

    for i in [0, 50, 150]:
        entry = all_windows[i]
        result = detect_quality(entry["window"])
        print(f"Window {i}  (from record: {entry['source_record']})")
        print(f"  -> {result}\n")