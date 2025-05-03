# hate-speech-detection
ML-based Hate Speech Detection Web App with Speech Input &amp; Dark UI

# 🛡️ Hate Speech Detection Web App

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)
![Status](https://img.shields.io/badge/status-Active-green)

> A real-time machine learning web app to detect hate speech in text using Flask, advanced UI, and speech-to-text capabilities.

---

## 🔍 Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect hate speech in user-provided text. It features:

- 🧠 ML Model trained on labeled dataset
- 🌐 Flask-based web app
- 🎤 Speech-to-text input (microphone)
- 🌙 Dark theme & modern UI
- ⚡ Real-time predictions with confidence score

---

## 🚀 Features

| Feature                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| 🔤 Text Input            | Enter any text to check for hate speech             |
| 🎙️ Speech-to-Text       | Convert spoken words to text for analysis           |
| 📊 Model Accuracy       | ~86% accuracy on real dataset                       |
| 🎨 Modern UI            | Dark themed, interactive, presentation-ready design |
| 🔐 REST API (Optional)  | Predict using external API requests (JSON)          |

---

## 📁 Project Structure

hate_speech_detection/
│
├── app.py # Flask backend
├── model.joblib # Trained ML model
├── cleaned_hatespeech.csv # Preprocessed dataset
├── templates/
│ └── index.html # Frontend HTML
├── static/
│ ├── style.css # Styling
│ └── script.js # JS for speech recognition
├── feature_extraction.py # Vectorization of text
├── train_model.py # Model training
├── predict.py # Inference script
├── requirements.txt # Dependencies
└── README.md # This file

# Create Virtual Enviornment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate   # On macOS/Linux

# Run the APP

python app.py

🧠 ML Model
Model Type: Logistic Regression
Libraries: scikit-learn, joblib, pandas, nltk
Dataset: Cleaned text labeled as Hate Speech or Normal

🖼️ UI/UX
Custom CSS with dark modern theme
Responsive design
Live confidence score output
Feedback loop on detection

📃 License
This project is licensed under the MIT License.

🤝 Acknowledgements
Scikit-learn
Flask
NLTK
HTML/CSS Community Designs
Your hard work and effort 💪
