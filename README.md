
# Critique's Sentiment

A sentiment analysis web app that predicts the sentiment of user-provided text reviews. This project is built using **Streamlit** and incorporates text preprocessing, vectorization, and machine learning models to classify reviews as positive or negative.

## Features
- **Real-time Sentiment Prediction**: Users can input reviews and instantly get a prediction of whether the sentiment is positive or negative.
- **Comprehensive Text Preprocessing**: Includes steps like lowercasing, stop word removal, lemmatization, and sentiment-aware tokenization.
- **Machine Learning Model**: Utilizes a trained sentiment analysis model for classification.
- **Vectorization**: Text data is transformed using TF-IDF vectorization to feed into the model.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chethanreddy10/Critique-s-sentiment.git
   cd Critique-s-sentiment

2. **Create a virtual environment (optional but recommended)**:

For GIT Bash Terminal:

  python -m venv venv
source venv/bin/activate  
On Windows, use 'venv\Scripts\activate'


3. **Install the dependencies**:

For GIT Bash Terminal:

  pip install -r requirements.txt
  

4. **Run the app**:

For GIT Bash Terminal:

  streamlit run app.py


  Usage
Once the app is running, open the local server link (usually http://localhost:8501) in your browser. Enter a text review into the input box, and the app will output whether the sentiment is positive or negative.

Requirements
Python 3.7+
Required Python packages are listed in requirements.txt.
Preprocessing Steps
Lowercasing text
Stop word removal
Lemmatization
Sentiment-aware tokenization
TF-IDF Vectorization
Model
The trained machine learning model used for this app is preloaded in the model/ directory and loaded when the app starts. It uses TF-IDF vectorized data for sentiment classification.

Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.



