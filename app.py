import streamlit as st
import pickle
import re
import string
import json
import os
import pandas as pd
import base64
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
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
    layout="wide"
)

BLUE = "#232368"
YELLOW = "#e4640c"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLANG_PATH = os.path.join(BASE_DIR, 'slang.json')
CSV_PATH = os.path.join(BASE_DIR, 'REVIEW KAI.csv')
IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'image.png')

DIVIDER_HTML = '<div class="divider"></div>'

load_dotenv(os.path.join(BASE_DIR, '.env'))
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')


@st.cache_resource
def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase: Client = init_supabase()


def get_banner_bg_css():
    if not os.path.exists(IMAGE_PATH):
        return ""
    with open(IMAGE_PATH, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return f"background-image: url('data:image/png;base64,{b64}');"


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {{
        background-color: #f5f6fa;
        color: #1a1a2e;
        font-family: 'Inter', sans-serif;
    }}

    .stApp > header {{ background-color: transparent; }}

    section[data-testid="stSidebar"] {{
        background-color: {BLUE} !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}

    /* ====== HERO BANNER ====== */
    .hero-banner {{
        position: relative;
        width: 100%;
        min-height: 420px;
        {get_banner_bg_css()}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 0 0 20px 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}

    .hero-overlay {{
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.55) 60%, rgba(35,35,104,0.7) 100%);
        z-index: 1;
    }}

    .hero-content {{
        position: relative;
        z-index: 2;
        width: 100%;
        max-width: 700px;
        padding: 3rem 2rem 2.5rem;
        text-align: center;
    }}

    .hero-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}

    .hero-subtitle {{
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }}

    /* ====== TEXTAREA ====== */
    div.stTextArea > div > div > textarea {{
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 2px solid #d1d5db !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        min-height: 100px !important;
    }}

    div.stTextArea > div > div > textarea:focus {{
        border-color: {BLUE} !important;
        box-shadow: 0 0 0 3px rgba(35, 35, 104, 0.15) !important;
    }}

    div.stTextArea > label {{
        color: #333333 !important;
        font-weight: 600 !important;
    }}

    /* ====== GENERAL ====== */
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        color: {BLUE};
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}

    .subtitle {{
        text-align: center;
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }}

    .divider {{
        height: 3px;
        background: linear-gradient(90deg, {YELLOW}, {BLUE}, {YELLOW});
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }}

    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {BLUE};
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }}

    /* ====== RESULT ====== */
    .result-positive {{
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }}

    .result-negative {{
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }}

    .label-positive {{ font-size: 1.6rem; font-weight: 700; color: #22c55e; }}
    .label-negative {{ font-size: 1.6rem; font-weight: 700; color: #ef4444; }}
    .emoji-big {{ font-size: 3rem; margin-bottom: 0.5rem; }}

    /* ====== INFO BOX ====== */
    .info-box {{
        background-color: #ffffff;
        border-left: 4px solid {YELLOW};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #333333;
        font-size: 0.95rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    /* ====== REVIEW LIST (no background) ====== */
    .review-link {{
        display: block;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.15rem;
        border-left: 3px solid {BLUE};
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        color: #333333;
    }}

    .review-link:hover {{
        border-left-color: {YELLOW};
        text-decoration: underline;
        text-decoration-color: {YELLOW};
        text-underline-offset: 3px;
    }}

    .review-link-positive {{ border-left-color: #22c55e; }}
    .review-link-negative {{ border-left-color: #ef4444; }}

    .review-text {{
        color: #333333;
        font-size: 0.92rem;
        line-height: 1.5;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    .badge-positive {{
        background-color: #dcfce7;
        color: #22c55e;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    .badge-negative {{
        background-color: #fee2e2;
        color: #ef4444;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    .footer {{
        text-align: center;
        color: #999999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ====== BUTTONS ====== */
    .stButton > button {{
        background: linear-gradient(135deg, {BLUE}, #1a1a5e) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {YELLOW}, #c75a0a) !important;
        box-shadow: 0 0 20px rgba(228, 100, 12, 0.4) !important;
        transform: translateY(-1px) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* Yellow "Lihat Semua Review" button */
    .btn-yellow-link {{
        display: block;
        text-align: center;
        margin-top: 1rem;
        background: {YELLOW};
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
        box-sizing: border-box;
    }}

    .btn-yellow-link:hover {{
        background: #c75a0a;
        box-shadow: 0 0 16px rgba(228, 100, 12, 0.5);
        transform: translateY(-1px);
        color: #ffffff !important;
        text-decoration: none;
    }}

    /* ====== DETAIL ====== */
    .detail-card {{
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }}

    .detail-label {{
        font-size: 0.85rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }}

    .detail-value {{
        font-size: 1rem;
        color: #333;
        line-height: 1.6;
    }}

    /* ====== STAT CARD ====== */
    .stat-card {{
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    .stat-number {{
        font-size: 2rem;
        font-weight: 800;
        color: {BLUE};
    }}

    .stat-label {{
        font-size: 0.85rem;
        color: #888;
        font-weight: 500;
    }}

    /* ====== SUCCESS TOAST ====== */
    .success-toast {{
        background-color: #dcfce7;
        border-left: 4px solid #22c55e;
        color: #166534;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_slang():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, 'model_indobert.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model_rf.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    return tfidf, rf_model


@st.cache_data
def load_reviews():
    df = pd.read_csv(CSV_PATH, sep=';')
    return df


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
    x_vec = tfidf.transform([processed])
    prediction = rf_model.predict(x_vec)[0]
    proba = rf_model.predict_proba(x_vec)[0]
    classes = rf_model.classes_
    prob_dict = dict(zip(classes, proba))
    return prediction, processed, prob_dict


def init_session_reviews():
    if 'reviews' not in st.session_state:
        st.session_state.reviews = []
        if supabase:
            try:
                resp = supabase.table('reviews').select('*').order('created_at', desc=True).limit(50).execute()
                for row in resp.data:
                    st.session_state.reviews.append({
                        'id': row['id'],
                        'text': row['review_text'],
                        'prediction': row['prediction'],
                        'prob_puas': row['prob_puas'],
                        'prob_tidak': row['prob_tidak_puas'],
                        'processed': row['processed_text'] or '',
                        'timestamp': datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')).strftime('%d/%m/%Y %H:%M'),
                    })
            except Exception:
                pass


def add_review_to_history(text, prediction, prob_dict, processed):
    prob_puas = prob_dict.get('Puas', 0) * 100
    prob_tidak = prob_dict.get('Tidak Puas', 0) * 100
    db_id = None

    if supabase:
        try:
            resp = supabase.table('reviews').insert({
                'review_text': text,
                'processed_text': processed,
                'prediction': prediction,
                'prob_puas': prob_puas,
                'prob_tidak_puas': prob_tidak,
            }).execute()
            if resp.data:
                db_id = resp.data[0]['id']
        except Exception:
            db_id = len(st.session_state.reviews) + 1

    if db_id is None:
        db_id = len(st.session_state.reviews) + 1

    entry = {
        'id': db_id,
        'text': text,
        'prediction': prediction,
        'prob_puas': prob_puas,
        'prob_tidak': prob_tidak,
        'processed': processed,
        'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
    }
    st.session_state.reviews.insert(0, entry)


init_session_reviews()
slang_dict = load_slang()
tfidf, rf_model = load_models()
df_reviews = load_reviews()


# ==================== PAGE ROUTING ====================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None

# Handle query param navigation from anchor clicks
qp = st.query_params
if 'detail' in qp:
    try:
        detail_id = int(qp['detail'])
        source = qp.get('source', 'home')
        st.session_state.selected_review_id = detail_id
        st.session_state.detail_source = source
        st.session_state.page = 'detail'
        del st.query_params['detail']
        if 'source' in st.query_params:
            del st.query_params['source']
    except (ValueError, TypeError):
        pass
elif 'page' in qp:
    target = qp['page']
    if target == 'history':
        st.session_state.page = 'history'
    del st.query_params['page']


def go_home():
    st.session_state.page = 'home'
    st.rerun()


def go_history():
    st.session_state.page = 'history'
    st.rerun()


def go_detail(review_id, source='home'):
    st.session_state.selected_review_id = review_id
    st.session_state.detail_source = source
    st.session_state.page = 'detail'
    st.rerun()


# ==================== HOME PAGE ====================
if st.session_state.page == 'home':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        # HERO BANNER with overlay
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-title">ANALISIS SENTIMEN</div>
                <div class="hero-subtitle">Review Kereta Api Indonesia &mdash; Puas atau Tidak Puas?</div>
                <div style="color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:0.3rem;">Analisis sentimen otomatis menggunakan Machine Learning untuk mengetahui kepuasan penumpang kereta api berdasarkan review.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Textarea on top of banner (inside the hero visual context)
        st.markdown('<div style="margin-top:-3.5rem; margin-bottom:1rem; position:relative; z-index:5;">', unsafe_allow_html=True)

        review_input = st.text_area(
            "Tulis review tentang pengalaman kereta api Anda:",
            placeholder="Contoh: pelayanannya ramah dan menyenangkan, kereta bersih dan nyaman...",
            label_visibility="collapsed",
            height=100,
            key="review_input_area"
        )

        if st.button("Analisis Sentimen", width="stretch", key="analyze_btn"):
            if review_input.strip():
                with st.spinner("Memproses..."):
                    prediction, processed, prob_dict = predict_sentiment(
                        review_input, tfidf, rf_model, slang_dict
                    )
                    add_review_to_history(review_input, prediction, prob_dict, processed)
                    st.toast("Review berhasil dianalisis! Klik review di daftar untuk melihat hasil.", icon="✅")
            else:
                st.warning("Silakan masukkan review terlebih dahulu.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(DIVIDER_HTML, unsafe_allow_html=True)

        # 5 Review Terakhir
        st.markdown(f'<div class="section-title">Review Terakhir</div>', unsafe_allow_html=True)

        recent_reviews = st.session_state.reviews[:5]
        if recent_reviews:
            for rev in recent_reviews:
                badge_class = 'badge-positive' if rev['prediction'] == 'Puas' else 'badge-negative'
                link_class = 'review-link-positive' if rev['prediction'] == 'Puas' else 'review-link-negative'
                truncated = rev['text'][:80]
                if len(rev['text']) > 80:
                    truncated += "..."
                st.markdown(f"""
                <a href="?detail={rev['id']}&source=home" target="_self" class="review-link {link_class}" style="text-decoration:none;">
                    <div class="review-text">{truncated}</div>
                    <div style="margin-top:0.3rem;">
                        <span class="{badge_class}">{rev['prediction']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">{rev['timestamp']}</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:1rem; text-align:center; color:#999; font-style:italic;">
                Belum ada review. Mulai analisis sentimen pertama Anda!
            </div>
            """, unsafe_allow_html=True)

        # Yellow button
        st.markdown(f"""
        <a href="?page=history" target="_self" class="btn-yellow-link">
            Lihat Semua Review
        </a>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="footer">
            Model: TF-IDF + Random Forest &bull; Data: Review KAI<br>
            Dibuat dengan Streamlit
        </div>
        """, unsafe_allow_html=True)


# ==================== HISTORY PAGE ====================
elif st.session_state.page == 'history':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        if st.button("Kembali ke Beranda", key="back_home_history"):
            go_home()

        st.markdown(f'<div class="main-title">Semua Review</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Riwayat analisis sentimen yang telah dilakukan</div>', unsafe_allow_html=True)
        st.markdown(DIVIDER_HTML, unsafe_allow_html=True)

        all_reviews = st.session_state.reviews
        if all_reviews:
            total = len(all_reviews)
            puas_count = sum(1 for r in all_reviews if r['prediction'] == 'Puas')
            tidak_count = total - puas_count

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total}</div>
                    <div class="stat-label">Total Review</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#22c55e;">{puas_count}</div>
                    <div class="stat-label">Puas</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#ef4444;">{tidak_count}</div>
                    <div class="stat-label">Tidak Puas</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

            for rev in all_reviews:
                badge_class = 'badge-positive' if rev['prediction'] == 'Puas' else 'badge-negative'
                link_class = 'review-link-positive' if rev['prediction'] == 'Puas' else 'review-link-negative'
                truncated = rev['text'][:100]
                if len(rev['text']) > 100:
                    truncated += "..."
                st.markdown(f"""
                <a href="?detail={rev['id']}&source=history" target="_self" class="review-link {link_class}" style="text-decoration:none;">
                    <div class="review-text">{truncated}</div>
                    <div style="margin-top:0.3rem;">
                        <span class="{badge_class}">{rev['prediction']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">{rev['timestamp']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">Puas: {rev['prob_puas']:.1f}%</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada review yang dianalisis. Kembali ke beranda untuk mulai analisis.")


# ==================== DETAIL PAGE ====================
elif st.session_state.page == 'detail':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        source_page = st.session_state.get('detail_source', 'home')
        if st.button("Kembali", key="back_from_detail"):
            if source_page == 'history':
                go_history()
            else:
                go_home()

        review_id = st.session_state.selected_review_id
        review_data = None
        for r in st.session_state.reviews:
            if r['id'] == review_id:
                review_data = r
                break

        if review_data:
            badge_class = 'badge-positive' if review_data['prediction'] == 'Puas' else 'badge-negative'

            if review_data['prediction'] == "Puas":
                st.markdown("""
                <div class="result-positive">
                    <div class="emoji-big">&#x1F604;</div>
                    <div class="label-positive">Puas</div>
                    <div style="color:#666666; margin-top:0.5rem;">Review ini menunjukkan kepuasan terhadap layanan kereta api.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <div class="emoji-big">&#x1F61E;</div>
                    <div class="label-negative">Tidak Puas</div>
                    <div style="color:#666666; margin-top:0.5rem;">Review ini menunjukkan ketidakpuasan terhadap layanan kereta api.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="detail-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <span class="{badge_class}" style="font-size:0.9rem; padding:4px 14px;">{review_data['prediction']}</span>
                    <span style="color:#999; font-size:0.85rem;">{review_data['timestamp']}</span>
                </div>
                <div class="detail-label">Review Asli</div>
                <div class="detail-value" style="margin-bottom:1.2rem;">{review_data['text']}</div>
                <div class="detail-label">Teks Setelah Preprocessing</div>
                <div class="detail-value" style="background:#f5f6fa; padding:0.8rem; border-radius:8px; font-family:monospace; font-size:0.9rem;">{review_data['processed']}</div>
            </div>
            """, unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#22c55e;">Probabilitas Puas</strong><br>
                    <span style="font-size:1.8rem; font-weight:700; color:#22c55e;">{review_data['prob_puas']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            with pc2:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#ef4444;">Probabilitas Tidak Puas</strong><br>
                    <span style="font-size:1.8rem; font-weight:700; color:#ef4444;">{review_data['prob_tidak']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Review tidak ditemukan.")
