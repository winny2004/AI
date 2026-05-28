import streamlit as st
import pickle
import re
import string
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="Analisis Sentimen Review KAI",
    page_icon="🚂",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #ff1a1a, #cc0000, #8b0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #cc0000, transparent);
        border: none;
        margin: 2rem 0;
    }

    .result-positive {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
        border: 2px solid #00cc44;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }

    .result-negative {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
        border: 2px solid #ff1a1a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }

    .label-positive {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00cc44;
    }

    .label-negative {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ff1a1a;
    }

    .emoji-big {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ff3333;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #cc0000;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #cccccc;
        font-size: 0.95rem;
        margin: 0.5rem 0;
    }

    .footer {
        text-align: center;
        color: #555555;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #222222;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    div.stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }

    div.stTextInput > div > div > input:focus {
        border-color: #cc0000 !important;
        box-shadow: 0 0 0 2px rgba(204, 0, 0, 0.3) !important;
    }

    div.stTextInput > label {
        color: #cccccc !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #cc0000, #8b0000) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a, #cc0000) !important;
        box-shadow: 0 0 20px rgba(204, 0, 0, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }
</style>
""", unsafe_allow_html=True)

SLANG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slang.json')

@st.cache_resource
def load_slang():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_models():
    with open('model_indobert.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model_rf.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return tfidf, rf_model

def case_folding(text):
    return text.lower()

def cleaning(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def normalisasi_kata(text, slang_dict):
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])

stop_words = set(stopwords.words('indonesian'))

def stopword_removal(tokens):
    return [word for word in tokens if word not in stop_words]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text, slang_dict):
    text = case_folding(text)
    text = cleaning(text)
    text = normalisasi_kata(text, slang_dict)
    tokens = word_tokenize(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)

def predict_sentiment(text, tfidf, rf_model, slang_dict):
    processed = preprocess_text(text, slang_dict)
    X_vec = tfidf.transform([processed])
    prediction = rf_model.predict(X_vec)[0]
    proba = rf_model.predict_proba(X_vec)[0]
    classes = rf_model.classes_
    prob_dict = dict(zip(classes, proba))
    return prediction, processed, prob_dict

st.markdown('<div class="main-title">ANALISIS SENTIMEN</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Review Kereta Api Indonesia &mdash; Positif atau Negatif?</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

slang_dict = load_slang()
tfidf, rf_model = load_models()

st.markdown('<div class="section-title">Masukkan Review</div>', unsafe_allow_html=True)

review_input = st.text_input(
    "Tulis review tentang pengalaman kereta api Anda:",
    placeholder="Contoh: pelayanannya ramah dan menyenangkan",
    label_visibility="collapsed"
)

if st.button("Analisis Sentimen", use_container_width=True):
    if review_input.strip():
        with st.spinner("Memproses..."):
            prediction, processed, prob_dict = predict_sentiment(
                review_input, tfidf, rf_model, slang_dict
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Hasil Analisis</div>', unsafe_allow_html=True)

        if prediction == "Puas":
            st.markdown("""
            <div class="result-positive">
                <div class="emoji-big">😄</div>
                <div class="label-positive">Puas</div>
                <div style="color:#999999; margin-top:0.5rem;">Review Anda menunjukkan kepuasan terhadap layanan kereta api.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
                <div class="emoji-big">😞</div>
                <div class="label-negative">Tidak Puas</div>
                <div style="color:#999999; margin-top:0.5rem;">Review Anda menunjukkan ketidakpuasan terhadap layanan kereta api.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            prob_puas = prob_dict.get('Puas', 0) * 100
            st.markdown(f"""
            <div class="info-box">
                <strong style="color:#00cc44;">Probabilitas Puas</strong><br>
                <span style="font-size:1.5rem; font-weight:700; color:#00cc44;">{prob_puas:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            prob_tidak = prob_dict.get('Tidak Puas', 0) * 100
            st.markdown(f"""
            <div class="info-box">
                <strong style="color:#ff1a1a;">Probabilitas Tidak Puas</strong><br>
                <span style="font-size:1.5rem; font-weight:700; color:#ff1a1a;">{prob_tidak:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Lihat teks yang telah diproses"):
            st.code(processed, language=None)
    else:
        st.warning("Silakan masukkan review terlebih dahulu.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.expander("Contoh Review"):
    examples = [
        ("pelayanannya ramah dan menyenangkan", "Puas"),
        ("kereta telat 2 jam dan tidak ada penjelasan", "Tidak Puas"),
        ("sangat puas dengan pelayanan petugas yang sopan", "Puas"),
        ("mengecewakan, AC rusak dan petugas tidak responsif", "Tidak Puas"),
        ("lumayan lah tapi kursinya agak sempit", "?"),
    ]
    for review, label in examples:
        st.markdown(f"""
        <div style="background:#1a1a1a; border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.5rem;
                     border-left:3px solid {'#00cc44' if label=='Puas' else '#ff1a1a' if label=='Tidak Puas' else '#666'};">
            <span style="color:#cccccc;">{review}</span>
            <span style="float:right; color:{'#00cc44' if label=='Puas' else '#ff1a1a' if label=='Tidak Puas' else '#666'};
                         font-weight:600;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Model: IndoBERT (TF-IDF + Random Forest) &bull; Data: Review KAI<br>
    Dibuat dengan Streamlit
</div>
""", unsafe_allow_html=True)
