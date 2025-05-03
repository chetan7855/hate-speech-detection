import joblib
import numpy as np
import pickle

# Load the saved model and vectorizer
model = joblib.load("model.joblib")
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def predict_hate_speech(text):
    """Predict whether the given text is hate speech or not."""
    # Transform text into feature vector
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized)[0]  # Probability scores

    # Interpret the result
    label_map = {0: "Not Hate Speech", 1: "Hate Speech"}
    result = label_map[prediction]

    return result, confidence

if __name__ == "__main__":
    while True:
        text = input("\nEnter text for hate speech detection (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break

        result, confidence = predict_hate_speech(text)
        print(f"\n🔹 Prediction: {result}")
        print(f"🔹 Confidence Scores: {confidence}\n")
