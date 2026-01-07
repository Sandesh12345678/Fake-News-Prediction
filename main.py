import numpy as np
import pandas as pd
import re
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Prediction",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Prediction System")
st.write("This app predicts whether a news article is **Real** or **Fake** using Machine Learning.")

# --------------------------------------------------
# Download stopwords (run once)
# --------------------------------------------------
nltk.download('stopwords')

# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Day4_Fake_News_Data.csv")
    df = df.fillna("")
    df["content"] = df["author"] + " " + df["title"]
    return df

news_dataset = load_data()

# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------
port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [
        port_stem.stem(word)
        for word in content
        if word not in stopwords.words('english')
    ]
    return ' '.join(content)

@st.cache_data
def preprocess_data(df):
    df["content"] = df["content"].apply(stemming)
    X = df["content"].values
    Y = df["label"].values
    return X, Y

X, Y = preprocess_data(news_dataset)

# --------------------------------------------------
# Vectorization & Model Training
# --------------------------------------------------
@st.cache_resource
def train_model(X, Y):
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vec, Y, test_size=0.2, stratify=Y, random_state=2
    )

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc = accuracy_score(Y_test, model.predict(X_test))

    return model, vectorizer, train_acc, test_acc

model, vectorizer, train_acc, test_acc = train_model(X, Y)

# --------------------------------------------------
# Show accuracy
# --------------------------------------------------
st.subheader("📊 Model Performance")
st.write(f"**Training Accuracy:** {train_acc:.2f}")
st.write(f"**Testing Accuracy:** {test_acc:.2f}")

# --------------------------------------------------
# User input
# --------------------------------------------------
st.subheader("📝 Enter News Text")

user_input = st.text_area(
    "Paste the news article here:",
    height=200
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_text = stemming(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)

        if prediction[0] == 0:
            st.success("✅ This news is **REAL**")
        else:
            st.error("🚨 This news is **FAKE**")
