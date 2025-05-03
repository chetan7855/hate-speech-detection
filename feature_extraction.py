import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset with correct dtype handling
df = pd.read_csv("cleaned_hatespeech.csv", low_memory=False)

# Ensure correct column names
if "clean_text" not in df.columns or "Label" not in df.columns:
    raise KeyError("🛑 Missing required columns: 'clean_text' or 'Label' in the dataset.")

# Drop potential header duplication issue
df = df[df["Label"] != "Label"]

# Convert text column to string and handle NaN values
df["clean_text"] = df["clean_text"].astype(str).fillna("")

# Convert labels to numeric, forcing errors='coerce' to handle bad data
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")

# Drop rows where Label conversion failed
df = df.dropna(subset=["Label"])

# Convert labels to integers
df["Label"] = df["Label"].astype(int)

# Extract text and labels
X_text = df["clean_text"]
y = df["Label"]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=3000)

# Transform text data
X = vectorizer.fit_transform(X_text)

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save features as .npz (for train_model.py)
np.savez_compressed("tfidf_features.npz", arr_0=X.toarray())

# Save features as .pkl (optional for flexibility)
with open("tfidf_features.pkl", "wb") as f:
    pickle.dump(X, f)

# Save labels
np.save("labels.npy", y)

print("✅ Feature extraction complete.")
print("🔹 Saved vectorizer as 'tfidf_vectorizer.pkl'")
print("🔹 Saved features as 'tfidf_features.npz' and 'tfidf_features.pkl'")
print("🔹 Saved labels as 'labels.npy'")
