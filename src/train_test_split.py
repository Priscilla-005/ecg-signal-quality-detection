from feature_extraction import feature_table
from sklearn.model_selection import train_test_split

# Separate features (X) from the label (y)
X = feature_table.drop(columns=["label"])
y = feature_table["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("\nTraining label counts:\n", y_train.value_counts())
print("\nTest label counts:\n", y_test.value_counts())