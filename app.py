import streamlit as st
import asyncio
import edge_tts
import os
import re
import shutil
import tempfile
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import pytesseract
try:
    from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
except ImportError:
    try:
        from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
    except ImportError:
        ImageClip = None
        AudioFileClip = None
        CompositeVideoClip = None

# Load environment variables
load_dotenv()

# --- TESSERACT AUTO-DETECTION ---
def find_tesseract():
    env_path = os.getenv("TESSERACT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    try:
        path = shutil.which("tesseract")
        if path:
            return path
    except:
        pass
    return None

TESSERACT_EXE = find_tesseract()
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

VOICE_MAPPINGS = {
    "English India (Male)": "en-IN-PrabhatNeural",
    "English India (Female)": "en-IN-NeerjaNeural",
    "English US (Male)": "en-US-ChristopherNeural",
    "English US (Female)": "en-US-JennyNeural",
    "Tamil India (Male)": "ta-IN-ValluvarNeural",
    "Tamil India (Female)": "ta-IN-PallaviNeural",
    "Telugu (Male)": "te-IN-MohanNeural",
    "Telugu (Female)": "te-IN-ShrutiNeural",
    "Kannada (Male)": "kn-IN-GaganNeural",
    "Kannada (Female)": "kn-IN-SapnaNeural",
    "Malayalam (Male)": "ml-IN-MidhunNeural",
    "Malayalam (Female)": "ml-IN-SobhanaNeural",
    "Hindi (Male)": "hi-IN-MadhurNeural",
    "Hindi (Female)": "hi-IN-SwararaNeural",
}

SPEED_MAP = {
    "0.25x": "-75%", "0.5x": "-50%", "0.75x": "-25%", "1x": "+0%",
    "1.25x": "+25%", "1.5x": "+50%", "2x": "+100%"
}

def ocr_via_api(image_bytes):
    try:
        url = "https://api.ocr.space/parse/Image"
        api_key = os.getenv("OCR_SPACE_API_KEY", "K8Scenario")
        files = {'file': image_bytes}
        data = {'apikey': api_key, 'language': 'eng', 'isOverlay': 'false', 'OCREngine': '2'}
        response = requests.post(url, files=files, data=data, timeout=30)
        result = response.json()
        if result.get("OCRExitCode") == 1:
            return "\n".join([line["Text"] for line in result.get("ParsedResults", [])])
        return f"API Error: {result.get('ErrorMessage', 'Unknown error')}"
    except Exception as e:
        return f"API Fallback Error: {str(e)}"

@st.cache_data(show_spinner=False)
def extract_text_from_image_cached(image_bytes):
    if TESSERACT_EXE:
        try:
            import io
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            if text.strip():
                return text.strip()
        except Exception:
            pass
    return ocr_via_api(image_bytes)

async def generate_voiceover_async(text, voice_id, rate, output_path):
    communicate = edge_tts.Communicate(text, voice_id, rate=rate)
    await communicate.save(output_path)

def create_video_production(image_path, audio_path, output_path):
    try:
        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path)

        if hasattr(image_clip, 'set_duration'):
            image_clip = image_clip.set_duration(audio_clip.duration)
        else:
            image_clip.duration = audio_clip.duration

        try:
            if hasattr(image_clip, 'resized'):
                image_clip = image_clip.resized(height=1920)
            else:
                image_clip = image_clip.resize(height=1920)
        except Exception:
            pass

        w, h = image_clip.size
        center_x = w // 2
        try:
            if hasattr(image_clip, 'crop'):
                image_clip = image_clip.crop(x1=center_x - 540, y1=0, x2=center_x + 540, y2=1920)
        except Exception:
            pass

        try:
            video = image_clip.set_audio(audio_clip)
        except AttributeError:
            try:
                video = CompositeVideoClip([image_clip])
                video = video.set_audio(audio_clip)
            except Exception:
                video = image_clip
                video.audio = audio_clip

        video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
        audio_clip.close()
        image_clip.close()
        return True
    except Exception as e:
        st.error(f"Video Assembly Error: {e}")
        return False

# --- UI OVERHAUL ---
st.set_page_config(page_title="Content Creating Studio", layout="wide", page_icon="🎬")

st.markdown("""
    <style>
    /* General Theme */
    .main { background-color: #0e1117; color: #ffffff; font-family: 'Inter', sans-serif; }

    /* Fix scrolling and overflow issues */
    [data-testid="stAppViewContainer"] {
        overflow-y: auto !important;
    }
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* Step Card Styling */
    .step-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e445e;
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(45deg, #ff4b4b, #ff8e8e);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,75,75,0.4);
    }

    /* Inputs */
    .stTextArea textarea { background-color: #0e1117 !important; color: #ffffff !important; border: 1px solid #3e445e !important; border-radius: 10px !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #0e1117 !important; color: #ffffff !important; border: 1px solid #3e445e !important; border-radius: 10px !important; }

    /* Headers */
    h1, h2, h3 { color: #ff4b4b !important; }
    </style>
""", unsafe_allow_html=True)

# Header Section
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("assets/jp.png", width=120)

with col_title:
    st.title("🎬 Content Creating Studio")
    st.markdown("#### Transform your images into professional educational reels in seconds.")

# PERSONAL BRANDING SECTION
st.markdown("---")
col_brand_l, col_brand_m, col_brand_r = st.columns([2, 1, 2])
with col_brand_m:
    st.image("assets/director.png", width=150)
    st.markdown("<p style='text-align: center;'><b>Studio Director</b></p>", unsafe_allow_html=True)
st.markdown("---")

# PROGRESS ROADMAP
st.markdown("---")
cols_road = st.columns(4)
with cols_road[0]: st.markdown("🟢 **1. Import**")
with cols_road[1]: st.markdown("⚪ **2. Script**")
with cols_road[2]: st.markdown("⚪ **3. Voice**")
with cols_road[3]: st.markdown("⚪ **4. Render**")
st.markdown("---")

# Main Workflow
col_left, col_right = st.columns([1.2, 1])

with col_left:
    # STEP 1: IMPORT
    st.markdown("### 📁 Step 1: Import Asset")
    with st.container():
        uploaded_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            st.image(uploaded_file, caption="Source Asset", use_container_width=True)
            if st.button("✨ Extract Text from Image"):
                with st.status("Analyzing Image...", expanded=True) as status:
                    image_bytes = uploaded_file.getvalue()
                    extracted = extract_text_from_image_cached(image_bytes)
                    if "OCR Error" in extracted or "API Error" in extracted:
                        st.error(extracted)
                    else:
                        st.session_state['extracted_text'] = extracted
                        status.update(label="Text Extracted!", state="complete", expanded=False)

    # STEP 2: SCRIPT
    st.markdown("### ✍️ Step 2: Refine Script")
    if 'extracted_text' not in st.session_state:
        st.session_state['extracted_text'] = ""

    script_text = st.text_area(
        "Edit your voice-over script here:",
        value=st.session_state['extracted_text'],
        height=300,
        label_visibility="collapsed"
    )

with col_right:
    # STEP 3: VOICE
    st.markdown("### 🎙️ Step 3: Voice Design")
    with st.container():
        voice_choice = st.selectbox("Select Voice Persona", options=list(VOICE_MAPPINGS.keys()))
        speed_choice = st.selectbox("Select Speech Rate", options=list(SPEED_MAP.keys()), index=3)

        st.info("💡 **Pro Tip:** Use 'English India Female' for a natural, friendly educational tone.")

    st.markdown("---")

    # STEP 4: RENDER
    st.markdown("### 🚀 Step 4: Produce Video")
    if st.button("🎬 Render Final MP4", type="primary"):
        if not uploaded_file or not script_text:
            st.warning("Please provide both an image and a script!")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with st.status("Production Pipeline active...", expanded=True) as status:
                        temp_img_path = os.path.join(tmpdir, "input.png")
                        temp_audio_path = os.path.join(tmpdir, "voice.mp3")
                        temp_video_path = os.path.join(tmpdir, "output.mp4")

                        status.write("📦 Packaging assets...")
                        with open(temp_img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        status.write("🎙️ Synthesizing neural voice...")
                        voice_id = VOICE_MAPPINGS[voice_choice]
                        rate = SPEED_MAP[speed_choice]
                        asyncio.run(generate_voiceover_async(script_text, voice_id, rate, temp_audio_path))

                        status.write("🎞️ Assembling 9:16 Reel...")
                        success = create_video_production(temp_img_path, temp_audio_path, temp_video_path)

                        if success:
                            status.update(label="Production Complete!", state="complete", expanded=False)
                            st.balloons()
                            with open(temp_video_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button("📥 Download Final MP4", data=video_bytes, file_name="inner_child_reel.mp4", mime="video/mp4")
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

st.divider()
st.caption("Content Creating Studio v1.1 | Hybrid OCR | Neural TTS | Auto-Assemble")
