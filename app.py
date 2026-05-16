import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Spam Filter", page_icon="🔒", layout="centered")

st.title("🔒 AI Message Spam Classifier")
st.write("Final Semester BSc Academic Project Demo")
st.write("---")

@st.cache_resource
def train_model():
    # Reads the dataset from your GitHub folder
    df = pd.read_csv('spam.csv', sep='\t', names=['label', 'message'], encoding='latin-1', on_bad_lines='skip')
    df = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    cv = CountVectorizer(stop_words='english')
    X_train_numeric = cv.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_numeric, y_train)
    return model, cv

model, cv = train_model()

user_input = st.text_area("Paste or type your message here:", placeholder="Enter SMS text...")

if st.button("Scan Message"):
    if user_input.strip() != "":
        numeric_input = cv.transform([user_input])
        prediction = model.predict(numeric_input)[0]
        
        if prediction == 'spam':
            st.error("⚠️ WARNING: This message looks like SPAM!")
        else:
            st.success("✅ SAFE: This message is regular HAM.")
    else:
        st.warning("Please enter some text first.")
