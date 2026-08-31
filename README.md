# ECG Signal Quality & Artifact Detection Using Machine Learning

## What this project does

This is a python and machine learning project that looks at short segments
of ECG signal and figures out what kind of physical activity the person
was doing when it was recorded — standing, walking, or jumping. Since
movement is the main thing that corrupts ECG signals in real wearable
devices, knowing the activity gives a solid, evidence-backed way to flag
how much motion artifact a given segment probably has.

## Why I built this

Wearable ECG monitors are great in theory - they let doctors keep an eye
on a patient's heart during normal daily life, not just during a short
clinic visit. The problem is that once someone starts moving around
(walking to work, climbing stairs, etc.), the signal gets messy, and
that messiness can either hide a real cardiac event or create a fake
alarm. I wanted to build something that tackles that specific problem:
given a short ECG window, can a model tell what kind of movement was
happening, and by extension, how trustworthy that segment probably is?

## The dataset

I used the **Motion Artifact Contaminated ECG Database (MACECGDB)** from
PhysioNet. It's a small but purpose-built dataset — one healthy 25-year-old
subject, recorded with a 4-channel electrode patch at 500 Hz, performing
27 short trials split evenly across standing, walking, and jumping (9 each,
across different patch placement angles).

It's licensed under the Open Data Commons Attribution License v1.0, which
just means it's freely usable as long as it's properly credited:

> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
> Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit,
> and PhysioNet: Components of a New Research Resource for Complex
> Physiologic Signals. *Circulation* 101(23):e215-e220.
> Dataset DOI: https://doi.org/10.13026/C2JP4G

## How the pipeline works
Raw ECG records (loaded with wfdb)
↓
Activity label pulled from the filename (s / w / j)
↓
Each record cut into 1-second windows (500 samples)
↓
9 features computed per window (7 statistical + 2 frequency-based)
↓
80/20 train/test split, stratified so classes stay balanced
↓
Two models trained and compared: Logistic Regression, Random Forest
↓
Evaluated with accuracy, confusion matrix, and per-class precision/recall
↓
Wrapped in a Streamlit app for a live, interactive demo

## The features

Seven of them are pretty standard signal statistics — mean, standard
deviation, variance, RMS, min, max, and range. These capture how
"jittery" or extreme a window looks.

The other two came from actually reading up on the problem: research on
ECG motion artifacts shows they tend to concentrate around 20–25 Hz,
while the heart's own signal sits closer to 5–10 Hz. So I added two
FFT-based features measuring how much of a window's energy falls in
the low-frequency band (a baseline wander indicator) versus the high-
frequency band (a motion-artifact indicator). Adding these actually
gave the biggest single accuracy jump in the whole project.

## How it performed

| What I tried | Model | Accuracy | Jump recall | What I learned |
|---|---|---|---|---|
| Starting point (2s windows) | Logistic Regression | 45% | 0.12 | Barely caught any jumps at all |
| Same data, different model | Random Forest | 55% | 0.62 | Random Forest handles the non-linear patterns much better |
| Shortened the windows to 1s | Random Forest | 57% | 0.73 | Shorter windows stop the brief jump event from getting "diluted" |
| Added frequency features | **Random Forest** | **64%** | 0.67 | Walking recall jumped from 0.33 to 0.53 — the biggest win here |

Worth putting in context: guessing randomly on 3 classes would land
around 33%. So 64% is roughly double chance — a real, if imperfect,
signal, not a coin flip.

## Limitations

- Trained on a single subject — may not generalize to other people
- Only 1 of 4 available electrode channels is used
- Walking is consistently the hardest class to classify correctly
- Artifact severity is inferred from the activity prediction, not
  measured independently per window (an R-peak-based independent
  check was explored but found infeasible with 1-second windows —
  see `src/experiments/detect_quality.py`)

## Future works

- Use all 4 ECG channels instead of just one
- Get data from more than one subject
- Revisit the independent quality-check idea with longer windows
- Try a CNN on the raw signal instead of hand-crafted features, given
  a bigger dataset
- Add proper cross-validation and hyperparameter tuning

## Tech stack

Python, `wfdb`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `joblib`


## Project layout

- `src/` — core pipeline (load → segment → extract features → split → train)
- `src/experiments/` — diagnostic scripts that shaped key decisions
  (tested and ruled out simpler threshold-based quality detection)
- `app.py` — Streamlit demo
- `results/` — saved model, confusion matrices, feature table

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```
