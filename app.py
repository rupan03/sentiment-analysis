import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Streamlit app
st.title("📱 Smartphone Review Sentiment Analyzer")
review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review.strip():
        cleaned = preprocess_text(review)
        vectorized = tfidf.transform([cleaned]).toarray()
        result = lr_model.predict(vectorized)[0]
        # Map numeric labels if necessary
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        result = label_map.get(result, result)
        if result == 'Positive':
            st.success(f"Predicted Sentiment: 😊 {result}")
        elif result == 'Negative':
            st.error(f"Predicted Sentiment: 😔 {result}")
        else:
            st.warning(f"Predicted Sentiment: 😐 {result}")
    else:
        st.warning("Please enter a review to analyze.")