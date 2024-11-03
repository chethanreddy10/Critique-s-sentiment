

# Critique's Sentiment


A sentiment analysis web app that predicts the sentiment of user-provided text reviews. 
This project is built using **Streamlit** and leverages comprehensive text preprocessing, TF-IDF vectorization, and machine learning models to classify reviews as positive or negative.

## Overview
Critique's Sentiment provides an accessible way to perform sentiment analysis on user-input text. The app uses preprocessing to clean and standardize text data, followed by vectorization and classification using a trained sentiment analysis model. With a simple interface, users can see instant sentiment results.

## Features
- **Real-time Sentiment Prediction**: Users receive instant feedback on whether a review is positive or negative.
- **Text Preprocessing Pipeline**: Detailed text processing steps such as lowercasing, stop word removal, lemmatization, and sentiment-aware tokenization.
- **ML Model for Classification**: A trained model(SVM) classifies text sentiment with high accuracy. 
- **TF-IDF Vectorization**: Converts text data into a format (alloactes text data in vector space as numerical data) that machine learning models can utilize effectively.

 Project Structure
```plaintext
Critique-s-sentiment/
├── app.py               # Main Streamlit app file
├── requirements.txt     # List of dependencies
├── model/               # Directory containing the trained model and vectorizer
├── preprocessing.py     # Preprocessing functions for text data
├── utils.py             # Utility functions for the app
└── README.md            # Project README file
```

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chethanreddy10/Critique-s-sentiment.git
   cd Critique-s-sentiment
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   ```

   - For **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - For **Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Usage
Once the app is running, open the local server link (usually http://localhost:8501) in your browser.
Enter a text review into the input box, and the app will output whether the sentiment is positive or negative.

## Requirements
- **Python 3.7+**
- Required Python packages are listed in `requirements.txt`.

## Preprocessing Steps
- Lowercasing text
- Stop word removal
- Lemmatization
- Sentiment-aware tokenization
- TF-IDF Vectorization

## Model
The trained machine learning model used for this app is preloaded in the `model/` directory  as pickle file and loaded when the app starts.
The dataset is trained on 2 models 1.Naive Bayes Multinomial classifier 2. Support Vector Machine. We picked SVM classifier as it gave us good accuracy and results.
It uses TF-IDF vectorized data which is fitted with the training data for sentiment classification.  So now when we give any New test input it will be tokenized and 
the vector space is already predetermined so it will get vectorized with respect to the inital fitted data.
