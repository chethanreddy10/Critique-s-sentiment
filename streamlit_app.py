import streamlit as st
# General libraries
import numpy as np
import pandas as pd
# NLP libraries
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import contractions
from bs4 import BeautifulSoup

# Download NLTK resources only once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer object
wnl = WordNetLemmatizer()

# Customizing the app style
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")


# Adding CSS styles to make it more attractive
st.markdown(
    """
    <style>
    /* Set background image */
    .main {
        background-image: url('https://www.link-to-your-background-image.com');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
    }

    .reportview-container {
        background: rgba(0, 0, 139, 0.7); /* Semi-transparent blue overlay */
        border-radius: 10px;
    }

    /* Text Styling */
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-size: 48px;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
    }
    h2 {
        color: #ffffff;
        text-align: center;
        font-size: 30px;
        font-family: 'Arial', sans-serif;
        margin-bottom: 30px;
    }

    /* Custom button */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6347;
    }

    /* Sentiment Result Styling */
    .stSuccess, .stError {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        animation: fadeIn 1s ease-in-out;
    }

    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* Footer styling */
    footer {
        font-size: 18px;
        color: white;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True
)


# Load vectorizer and model outside the submit button to avoid reloading
@st.cache_data
def load_model_and_vectorizer():
    vectorizer = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('nb_model2.pkl', 'rb'))
    return vectorizer, model


vectorizer, model = load_model_and_vectorizer()


# Helper function to map POS tag to wordnet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


# Clean data function with caching
@st.cache_data
def clean_data(text):
    # Convert to lowercase
    text = text.lower()

    # Tie "not" with the next word to retain the negative sentiment
    text = re.sub(r'\bnot\b \b\w+\b', lambda x: x.group().replace(' ', '_'), text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove special characters, numbers, and punctuations
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)

    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove HTML tags
    text = BeautifulSoup(text, 'lxml').get_text()

    # Remove any emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove any mentions
    text = re.sub(r'@\w+', '', text).strip()

    # Remove repeated characters
    text = re.sub(r'(.)\1+', r'\1\1', text)

    return text.strip()


# Lemmatization function with caching
@st.cache_data
def lemmatize(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in set(stopwords.words('english'))]
    pos_tagged = nltk.pos_tag(filtered_words)
    lemmatized_words = [wnl.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tagged]
    return " ".join(lemmatized_words)


# Create the title of the app with emoji
st.title('🎬 Movie Review Sentiment Analysis 🎥')

# Subheader for the app
st.subheader("Analyze your movie review's sentiment!")

# Input for user review with a text area for more space
review = st.text_area("Enter your Movie Review", placeholder="Type your review here...")

# Predict sentiment button
if st.button("Predict Sentiment 🚀"):
    with st.spinner('Analyzing your review...'):
        # Clean and lemmatize the input text
        cleaned_data = clean_data(review)
        lemmatized_data = lemmatize(cleaned_data)

        # Predict sentiment
        prediction = model.predict(vectorizer.transform([lemmatized_data]))

        # Display sentiment result with emoji
        if prediction == 0:
            st.error("😞 The Review is NEGATIVE!")
        else:
            st.success("😊 The Review is POSITIVE!")

# Footer for the app
st.markdown("<br><hr><footer>THANK YOU FOR YOUR REVIEW!</footer><hr>", unsafe_allow_html=True)
