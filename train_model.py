import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load dataset safely
df = pd.read_csv("cleaned_hatespeech.csv", low_memory=False)

# ✅ Ensure 'Label' column exists
if "Label" not in df.columns:
    raise KeyError("🛑 'Label' column is missing in the dataset. Check CSV structure.")

# ✅ Remove non-numeric values in 'Label'
df = df[df["Label"].astype(str).str.isnumeric()]

# ✅ Convert to integer
df["Label"] = df["Label"].astype(int)

# ✅ Print unique labels to check if data is clean
print("Unique labels:", df["Label"].unique())

# ✅ Load vectorized features
X = np.load("tfidf_features.npz")["arr_0"]
y = df["Label"].values

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save Model
joblib.dump(model, "model.joblib")
print("✅ Model saved as 'model.joblib'")
