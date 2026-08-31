from train_test_split import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Create the model
model = LogisticRegression(max_iter=1000)

# Train it on the training data
model.fit(X_train, y_train)

# Use it to predict on the test data (which it has never seen)
y_pred = model.predict(X_test)

print("Predictions on test set:", list(y_pred))
print("Actual labels on test set:", list(y_test))

#confusion matrix and proper accuracy metrices
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["standing", "walking", "jumping"]))

from sklearn.ensemble import RandomForestClassifier

# Create and train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["standing", "walking", "jumping"]))

# Save Logistic Regression confusion matrix as an image
disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred),
    display_labels=["standing", "walking", "jumping"]
)
disp_lr.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("results/confusion_matrix_logistic_regression.png")
plt.close()

# Save Random Forest confusion matrix as an image
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_rf),
    display_labels=["standing", "walking", "jumping"]
)
disp_rf.plot()
plt.title("Random Forest - Confusion Matrix")
plt.savefig("results/confusion_matrix_random_forest.png")
plt.close()

print("\nConfusion matrix images saved to results/ folder.")
import joblib

joblib.dump(rf_model, "results/final_rf_model.pkl")
print("Model saved to results/final_rf_model.pkl")

# Calculate baseline "clean" statistics from standing-only windows in training data
standing_baseline = X_train[y_train == 0]

baseline_stats = {
    "std_mean": standing_baseline["std"].mean(),
    "std_std": standing_baseline["std"].std(),
    "rms_mean": standing_baseline["rms"].mean(),
    "rms_std": standing_baseline["rms"].std(),
}

import json
with open("results/baseline_stats.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)

print("Baseline stats saved:", baseline_stats)