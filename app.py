import streamlit as st
import pickle

# Load Model and Vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Sentiment Analysis App")
review = st.text_area("Enter a review:")

if st.button("Predict"):
    X_test_example_tfidf = tfidf.transform([review])
    prediction = model.predict(X_test_example_tfidf)[0]
    probability = model.predict_proba(X_test_example_tfidf)[0].max()
    sentiment_label = "Positive" if prediction == 1 else "Negative"
    
    st.write(f"Predicted Sentiment: **{sentiment_label}**")
    st.write(f"Confidence: **{round(probability, 2)}**")
