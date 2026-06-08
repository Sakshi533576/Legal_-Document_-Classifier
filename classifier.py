import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Legal Document Classifier",
    page_icon="⚖️",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS + BACKGROUND IMAGE
# -----------------------------
st.markdown(
    """
    <style>

    /* Full App Background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1589829545856-d10d557cf95f");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    /* Dark Overlay */
    .main::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.55);
        z-index: -1;
    }

    /* Title Styling */
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #dfe6e9;
        margin-bottom: 30px;
    }

    /* Glassmorphism Card */
    .card {
        background: rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
    }

    /* Text Area */
    textarea {
        background-color: rgba(0,0,0,0.7) !important;
        color: white !important;
        font-size: 16px !important;
        border-radius: 12px !important;
    }

    textarea::placeholder {
        color: #cccccc !important;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        background: linear-gradient(to right, #0072ff, #00c6ff);
    }

    /* Prediction Result */
    .result-box {
        text-align: center;
        background: rgba(0,0,0,0.45);
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">⚖️ Legal Document Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">📄 AI-based Legal Text Categorization System</div>',
    unsafe_allow_html=True
)

# -----------------------------
# DATASET
# -----------------------------
data = {
    "text": [
        "This agreement is made between two parties for sale of property",
        "The court hereby orders the defendant to appear",
        "This affidavit confirms the statement given by witness",
        "This contract is valid for one year between company and client",
        "The judgement was passed in favor of the plaintiff",
        "This notice is issued for violation of rules",
        "This contract is signed between buyer and seller",
        "Service agreement valid for one year between two parties",
        "The court has passed judgement in this civil case",
        "Final judgement delivered by district court",
        "This is a legal notice for rent payment default",
        "Notice issued for violation of property rules"
    ],
    "label": [
        "Agreement",
        "Court Order",
        "Affidavit",
        "Contract",
        "Judgement",
        "Notice",
        "Contract",
        "Agreement",
        "Judgement",
        "Judgement",
        "Notice",
        "Notice"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# CLEAN TEXT FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# -----------------------------
# MODEL TRAINING
# -----------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# EMOJI MAP
# -----------------------------
emoji_map = {
    "Agreement": "🤝",
    "Court Order": "🏛️",
    "Affidavit": "📝",
    "Contract": "📑",
    "Judgement": "⚖️",
    "Notice": "📢"
}

# -----------------------------
# INPUT UI
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### ✍️ Enter Legal Document Text Below")

user_input = st.text_area(
    "📄 Legal Text Input",
    height=180,
    placeholder="e.g. This contract is made between buyer and seller..."
)

if st.button("🔍 Predict Category"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some legal text first!")

    else:
        cleaned = clean_text(user_input)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        result = prediction[0]

        emoji = emoji_map.get(result, "📄")

        # Prediction Result
        st.markdown(
            f"""
            <div class="result-box">
                <h2>🎯 Predicted Category</h2>
                <h1>{result}</h1>
                <div style="font-size:90px;">{emoji}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <br><br>
    <center style="color:white;">
        ⚡ Built with Streamlit | ⚖️ Legal AI Project
    </center>
    """,
    unsafe_allow_html=True
)



