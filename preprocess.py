import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset and check column names
df = pd.read_csv("hatespeechdetection.csv")
print("✅ Columns in dataset:", df.columns)

# Use the correct column for text
TEXT_COLUMN = 'Content'  # ✅ Corrected based on your dataset

# Check if the column exists
if TEXT_COLUMN not in df.columns:
    raise KeyError(f"❌ Column '{TEXT_COLUMN}' not found! Use one of these instead: {list(df.columns)}")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

# Apply preprocessing to the 'content' column
df['clean_text'] = df[TEXT_COLUMN].apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_hatespeech.csv", index=False)

print("✅ Text preprocessing complete! Cleaned data saved as 'cleaned_hatespeech.csv'.")
