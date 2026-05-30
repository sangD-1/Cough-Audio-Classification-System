import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Load model
model = load_model("cough_cnn_model.h5")

# Page config
st.set_page_config(page_title="Cough Detection", layout="wide")

# 🌌 Background from web
background_url = "https://images.unsplash.com/photo-1559757175-5700dde675bc"
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        background: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 15px;
    }}

    .stButton>button {{
        background-color: #00c853;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ AI Cough Detection App")
st.write("Upload an audio file to analyze cough presence using deep learning.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])


def audio_to_image(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(spec_db, sr=sr)
    plt.axis('off')

    temp_img = "temp.png"
    plt.savefig(temp_img, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(temp_img).convert("RGB")
    img = img.resize((128, 128))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    os.remove(temp_img)

    return img, audio, sr, spec_db

# 🚀 Prediction
if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        img, audio, sr, spec_db = audio_to_image(tmp_path)

        prediction = model.predict(img)[0][0]
        confidence = float(prediction * 100)

        col1, col2 = st.columns(2)

        # Result
        with col1:
            st.subheader("🧠 Prediction Result")
            if prediction > 0.5:
                st.error(f"😷 Cough Detected ({confidence:.2f}%)")
                st.progress(min(int(confidence), 100))
            else:
                st.success(f"✅ No Cough ({100-confidence:.2f}%)")
                st.progress(min(int(100-confidence), 100))

            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        # 📊 Additional Data
        with col2:
            st.subheader("📊 Audio Insights")

            duration = len(audio) / sr
            st.write(f"**Duration:** {duration:.2f} sec")
            st.write(f"**Sample Rate:** {sr} Hz")
            st.write(f"**Audio Length:** {len(audio)} samples")

        # 📈 Waveform + 🎨 Spectrogram (smaller, side-by-side)
        colw1, colw2 = st.columns(2)

        with colw1:
            st.subheader("📈 Waveform")
            fig_wave, ax = plt.subplots(figsize=(4,2))
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_title("Waveform", fontsize=8)
            ax.tick_params(labelsize=6)
            st.pyplot(fig_wave)

        with colw2:
            st.subheader("🎨 Mel Spectrogram")
            fig_spec, ax = plt.subplots(figsize=(4,2))
            librosa.display.specshow(spec_db, sr=sr, ax=ax)
            ax.set_title("Spectrogram", fontsize=8)
            ax.tick_params(labelsize=6)
            st.pyplot(fig_spec)

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        os.remove(tmp_path)

# Footer
st.markdown("---")
