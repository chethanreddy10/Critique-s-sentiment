import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

# Download NLTK resources only once
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
wnl = WordNetLemmatizer()

# Set up Streamlit page
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

# Load vectorizer and model
@st.cache_resource  # Cached only once for all users
def load_model_and_vectorizer():
    vectorizer = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('svm.pkl', 'rb'))
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# Map POS tags for lemmatization
def get_wordnet_pos(tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(tag[0], wordnet.NOUN)

# Clean data function
@st.cache_data
def clean_data(text):
    text = text.lower()
    text = re.sub(r'(http\S+|www\S+|\@\w+|\#)', '', text)  # Remove URLs, @, and hashtags
    text = re.sub(r'\bnot\b \b\w+\b', lambda x: x.group().replace(' ', '_'), text)  # "not" modifier
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags directly with regex
    text = re.sub(r'\W|\d', ' ', text)  # Remove special chars and digits
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Remove repeated characters
    return re.sub(r'\s+', ' ', text).strip()

# Lemmatize text
@st.cache_data
def lemmatize(text):
    tokens = [wnl.lemmatize(word, get_wordnet_pos(pos))
              for word, pos in nltk.pos_tag(word_tokenize(text))
              if word.lower() not in stop_words]
    return " ".join(tokens)

# Updated CSS styling for a new color scheme
st.markdown(
    """
 <style>
    /* Background styling */
    .main {
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
        background-size: cover;
        padding: 20px;
    }
    /* Card and text styling */
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 2rem;
    }
    h1 {
        color: #ff6b6b;
        text-align: center;
        font-size: 50px;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000;
    }
    h2 {
        color: #f5f5f5;
        text-align: center;
        font-size: 30px;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff8a8a;
    }
    /* Sentiment result styling */
    .stAlert {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        animation: fadeIn 1s ease-in-out;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    footer {
        font-size: 18px;
        color: #ff6b6b;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎬 Movie Review Sentiment Analysis 🎥")
st.subheader("Analyze your movie review's sentiment!")

# User input
review = st.text_area("Enter your Movie Review", placeholder="Type your review here...")

# Predict sentiment
if st.button("Predict Sentiment 🚀"):
    with st.spinner('Analyzing your review...'):
        # Clean and lemmatize the input text
        cleaned_data = clean_data(review)
        lemmatized_data = lemmatize(cleaned_data)

        # Predict sentiment
        prediction = model.predict(vectorizer.transform([lemmatized_data]))[0]

        # Set sentiment message based on prediction
        sentiment = " The Review is POSITIVE!" if prediction == 1 else " The Review is NEGATIVE!"

        # Display result based on sentiment
        if prediction == 1:
            st.success(sentiment)
        else:
            st.error(sentiment)

# Footer
st.markdown("<br><footer>THANK YOU FOR USING OUR APP!</footer>", unsafe_allow_html=True)
