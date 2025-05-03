# save_model.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse
import numpy as np

# Load features and labels
X = scipy.sparse.load_npz("tfidf_features.npz")
y = np.load("labels.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model saved as 'model.joblib'")
