import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load your data and model
data = pd.read_csv('Malicious URL.csv')
bundle = joblib.load("rf_bundle.joblib")
model = bundle["model"]
label_encoder = bundle["label_encoder"]

# Import your feature extraction function
from model import extract_features_from_url  # Replace with your actual file name

# Create features from URLs
print("Extracting features...")
features = []
for url in data['url']:
    features.append(extract_features_from_url(url))

X = np.array(features)
y = label_encoder.transform(data['type'])

# Split into train/test (use same random state as your training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Test the model
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test accuracy
accuracy = (y_pred == y_test).mean()
print(f"\nTest Accuracy: {accuracy:.4f}")