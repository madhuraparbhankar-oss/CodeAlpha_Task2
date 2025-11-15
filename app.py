import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ================================
# Professional Handwritten Character Recognition App
# ================================

# Page config - MUST BE FIRST!
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


# Header
st.markdown('<h1 class="title">✍️ AI Handwriting Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform handwritten characters into digital text with cutting-edge AI technology</p>', unsafe_allow_html=True)

# 1️⃣ Load Model with error handling
@st.cache_resource
def load_model():
    try:
        model_path = r"enter_your_file_path\best_model.keras"
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if error:
    st.error(f"❌ Model loading failed: {error}")
    st.info("Please check if the model file exists at the specified path.")
    st.stop()

# Feature Section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center;">🎯</h3>
        <h4 style="text-align: center; color: #667eea;">High Accuracy</h4>
        <p style="text-align: center; color: #666;">98%+ recognition rate</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center;">⚡</h3>
        <h4 style="text-align: center; color: #667eea;">Instant Results</h4>
        <p style="text-align: center; color: #666;">Real-time processing</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="text-align: center;">🔒</h3>
        <h4 style="text-align: center; color: #667eea;">Secure</h4>
        <p style="text-align: center; color: #666;">Privacy guaranteed</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2️⃣ File Uploader
st.markdown("### 📤 Upload Your Image")
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "png", "jpeg"],
    help="Upload a clear image of a handwritten character (A-Z)"
)

# 3️⃣ Preprocess Function
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert("L")
    img = np.array(img)
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    # Invert if white background
    if np.mean(img) > 127:
        img = 255 - img
    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# 4️⃣ Prediction Logic
if uploaded_file is not None:
    col_img, col_result = st.columns([1, 1])
    
    with col_img:
        st.markdown("### 🖼️ Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col_result:
        st.markdown("### 🔍 Processed View")
        processed = preprocess_image(image)
        st.image(processed[0], use_column_width=True, clamp=True)
    
    # Predict
    with st.spinner("🧠 Analyzing handwriting..."):
        preds = model.predict(processed, verbose=0)
        pred_label = np.argmax(preds)
        confidence = np.max(preds)
        
        # Convert to character (A-Z for digits 0-25)
        if pred_label < 26:
            predicted_char = chr(pred_label + 65)
        else:
            predicted_char = str(pred_label)
    
    # Display results in a beautiful card
    st.markdown(f"""
    <div class="result-card">
        <h2 style="margin: 0;">🎉 Recognition Complete!</h2>
        <div class="predicted-char">{predicted_char}</div>
        <div class="confidence-text">Confidence: {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence bar
    st.markdown("### 📊 Confidence Level")
    st.progress(float(confidence))
    
    # Top 3 predictions
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    st.markdown("### 🏆 Top 3 Predictions")
    
    for i, idx in enumerate(top3_idx):
        char = chr(idx + 65) if idx < 26 else str(idx)
        prob = preds[0][idx]
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.markdown(f"**#{i+1}**")
        with col2:
            st.progress(float(prob))
        with col3:
            st.markdown(f"**{char}** ({prob:.1%})")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an image to get started")
    
    st.markdown("### 💡 Tips for Best Results:")
    st.markdown("""
    - ✅ Use clear, well-lit images
    - ✅ Write characters in the center
    - ✅ Use dark ink on light background
    - ✅ Avoid blurry or low-resolution images
    - ✅ Single character per image works best
    """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Built with ❤️ using</p>
    <p style="font-size: 1.3rem; font-weight: 600;">
        <span style="color: #FF4B4B;">Streamlit</span> × 
        <span style="color: #FF6F00;">TensorFlow</span> × 
        <span style="color: #667eea;">Deep Learning</span>
    </p>
</div>

""", unsafe_allow_html=True)
