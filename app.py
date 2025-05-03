from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import speech_recognition as sr

app = Flask(__name__)

# Load Model & Vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_hate_speech(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][1]  # Probability of hate speech
    return prediction, probability

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    prediction, probability = predict_hate_speech(text)
    return jsonify({
        "text": text,
        "prediction": int(prediction),
        "hate_speech_probability": float(probability)
    })

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Speech recognition service unavailable"}), 500

if __name__ == "__main__":
    app.run(debug=True)
