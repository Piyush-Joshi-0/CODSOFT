import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Custom function to load and preprocess data
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load the TF-IDF Vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the Naive Bayes Model
with open('model.pkl', 'rb') as f:
    nb_classifier = pickle.load(f)

# Streamlit app
def main():
    st.title('Movie Genre Classification')

    # User input for movie plot
    user_input = st.text_area('Enter the plot of the movie:')

    if st.button('Classify'):
        if user_input.strip() == '':
            st.error('Please enter a movie plot.')
        else:
            # Preprocess user input
            processed_input = preprocess_text(user_input)
            # Vectorize input
            input_vector = tfidf.transform([processed_input]).toarray()
            # Predict genre
            prediction = nb_classifier.predict(input_vector)
            # Reverse mapping for genre
            genre_mapper = {0: 'other', 1: 'action', 2: 'adventure', 3: 'comedy',
                            4: 'drama', 5: 'horror', 6: 'romance', 7: 'sci-fi',
                            8: 'thriller', 9: 'documentary', 10: 'adult'}
            genre = genre_mapper[prediction[0]]
            st.success(f'The predicted genre of the movie is: {genre}')

if __name__ == '__main__':
    main()
