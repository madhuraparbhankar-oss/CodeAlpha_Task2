import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ================================
# AI Handwritten Character Recognition
# ================================

# ----------------- STYLES -----------------
st.markdown("""
<style>
/* 🌈 Dynamic gradient black background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0d0d0d 20%, #000000 100%);
    background-image: linear-gradient(135deg, #0d0d0d 10%, #0a002e 40%, #16003b 60%, #001f3f 90%);
    background-size: 300% 300%;
    animation: gradientShift 15s ease infinite;
    color: white;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 🧱 Main container */
.main-container {
    background: rgba(25, 25, 25, 0.85);
    padding: 3rem;
    border-radius: 25px;
    box-shadow: 0 0 40px rgba(0, 245, 212, 0.15);
    margin: 2rem auto;
    max-width: 900px;
    backdrop-filter: blur(10px);
}

/* 💫 Title */
.title {
    color: #ffffff;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    text-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
    letter-spacing: 1px;
}

/* ✨ Subtitle heading */
.subtitle {
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 1rem;
    margin-bottom: 3rem;
    text-shadow: 0 0 15px rgba(0,245,212,0.4);
}

/* ⚡ Feature grid */
.feature-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1.5rem;
}

/* 🌈 Feature cards */
.feature-card {
    flex: 1 1 260px;
    color: white;
    padding: 1.8rem;
    border-radius: 18px;
    text-align: center;
    transition: all 0.4s ease;
    cursor: pointer;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}
.feature-card:nth-child(1) {
    box-shadow: 0 0 25px rgba(255,110,199,0.3);
    border-color: rgba(255,110,199,0.4);
}
.feature-card:nth-child(2) {
    box-shadow: 0 0 25px rgba(0,245,212,0.3);
    border-color: rgba(0,245,212,0.4);
}
.feature-card:nth-child(3) {
    box-shadow: 0 0 25px rgba(155,93,229,0.3);
    border-color: rgba(155,93,229,0.4);
}
.feature-card:hover {
    transform: translateY(-6px) scale(1.05);
    box-shadow: 0 0 40px rgba(255,255,255,0.25);
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.15));
}

/* 🖼️ Upload section */
.upload-section {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #ffea00;
    padding: 2rem;
    border-radius: 20px;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(255, 234, 0, 0.2);
}
.upload-section:hover {
    background: rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(255, 234, 0, 0.4);
}

/* 🌟 Yellow text for upload + tips */
.upload-section label,
.upload-section p,
.upload-section span,
h3, .tips-text {
    color: #ffea00 !important;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(255, 234, 0, 0.4);
}

/* 🔮 Result card */
.result-card h2 {
    color: #ffea00 !important; /* bright yellow */
    font-weight: 800;
    text-shadow: 0 0 15px rgba(255, 234, 0, 0.6);
}

.predicted-char {
    font-size: 6rem;
    font-weight: 900;
    text-shadow: 0 0 30px rgba(255,255,255,0.3);
}

.confidence-text {
    font-size: 1.5rem;
    opacity: 0.95;
}

/* 🎨 Buttons */
.stButton>button {
    background: linear-gradient(135deg, #9b5de5, #00f5d4, #fee440);
    color: black;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 40px;
    font-weight: 700;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px rgba(255,255,255,0.4);
}

/* 🌈 Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #9b5de5, #00f5d4, #fee440);
}

/* 🚫 Hide Streamlit branding */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# ----------------- HEADER -----------------
st.markdown('<h1 class="title">✍️ AI Handwriting Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform handwritten characters into digital text with cutting-edge AI</p>', unsafe_allow_html=True)


# ----------------- LOAD MODEL SAFELY -----------------
@st.cache_resource
def load_character_model():
    model_path = os.path.join(os.getcwd(), "best_model.keras")

    if not os.path.exists(model_path):
        return None  # DO NOT USE streamlit functions here!

    return tf.keras.models.load_model(model_path)


model = load_character_model()

# If missing — show message outside cache function
if model is None:
    st.error("❌ Model file 'best_model.keras' not found in repository.")
    st.info("Upload the model in the same folder as this app file.")
    st.stop()


# ----------------- FEATURES -----------------
col1, col2, col3 = st.columns(3)
col1.markdown("<h3>🎯 High Accuracy</h3>", unsafe_allow_html=True)
col2.markdown("<h3>⚡ Instant Results</h3>", unsafe_allow_html=True)
col3.markdown("<h3>🔒 Secure</h3>", unsafe_allow_html=True)


# ----------------- UPLOAD SECTION -----------------
st.markdown("### 📤 Upload Your Image")
uploaded_file = st.file_uploader("Upload a handwritten character", type=["png", "jpg", "jpeg"])


# ----------------- PREPROCESS IMAGE -----------------
def preprocess_image(image):
    img = image.convert("L")
    img = img.resize((28, 28))
    img = np.array(img)

    if np.mean(img) > 127:
        img = 255 - img

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


# ----------------- PROCESS + PREDICT -----------------
if uploaded_file:
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("### 🖼️ Original")
        image = Image.open(uploaded_file)
        st.image(image)

    with col_res:
        st.markdown("### 🔧 Processed")
        processed = preprocess_image(image)
        st.image(processed[0], clamp=True)

    with st.spinner("🧠 Recognizing character..."):
        preds = model.predict(processed)
        pred = np.argmax(preds)
        confidence = np.max(preds)

        # A–Z mapping
        predicted_char = chr(pred + 65) if pred < 26 else str(pred)

    st.markdown(
        f"""
        <h2 style='color:#ffea00;'>🎉 Result: {predicted_char}</h2>
        <h4>Confidence: {confidence:.2%}</h4>
        """,
        unsafe_allow_html=True
    )

    # Top 3 predictions
    st.markdown("### 🏆 Top 3 Predictions")
    top3 = np.argsort(preds[0])[-3:][::-1]
    for r, idx in enumerate(top3):
        label = chr(idx + 65) if idx < 26 else str(idx)
        st.write(f"**#{r+1}: {label}** — {preds[0][idx]:.2%}")

else:
    st.info("👆 Upload a character image to continue.")

