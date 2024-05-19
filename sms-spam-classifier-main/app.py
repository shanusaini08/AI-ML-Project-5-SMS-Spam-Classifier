import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and description
st.title("Email/SMS Spam Classifier")
st.write("This app classifies whether an input message is Spam or Not Spam.")

# Input text area for user message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    if input_sms:
        # Preprocess 
        transformed_sms = transform_text(input_sms)
        # Vectorization
        vector_input = tfidf.transform([transformed_sms])
        # Prediction
        result = model.predict(vector_input)[0]

        # Display prediction result
        st.subheader("Prediction:")
        if result == 1:
            st.success("This message is classified as Spam.")
        else:
            st.success("This message is Not Spam.")

        # Show prediction confidence
        st.subheader("Prediction Confidence:")
        accuracy = model.predict_proba(vector_input).max() * 100
        st.info(f"The model is {accuracy:.2f}% confident in its prediction.")

    else:
        st.warning("Please enter a message for prediction.")

# Footer with credits
st.markdown("Made By Shanu Saini")
