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

    /* Mobile Friendly adjustments */
    @media (max-width: 768px) {
        .studio-header { padding: 10px !important; }
        .studio-header h1 { font-size: 1.5rem !important; }
        .step-card { padding: 15px !important; }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
    }

    /* Fix scrolling and overflow issues */
    [data-testid="stAppViewContainer"] {
        overflow-y: auto !important;
    }
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }

    /* Step Card Styling */
    .step-card {
        background-color: #1e2130;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #3e445e;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Custom Header Styling */
    .studio-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(180deg, #1e2130 0%, #0e1117 100%);
        border-radius: 0 0 30px 30px;
        margin-bottom: 30px;
        border-bottom: 2px solid #ff4b4b;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(45deg, #ff4b4b, #ff8e8e);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,75,75,0.4);
        background: linear-gradient(45deg, #ff6b6b, #ffa8a8);
    }

    /* Inputs */
    .stTextArea textarea { background-color: #0e1117 !important; color: #ffffff !important; border: 1px solid #3e445e !important; border-radius: 12px !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #0e1117 !important; color: #ffffff !important; border: 1px solid #3e445e !important; border-radius: 12px !important; }

    /* Progress Bar Styling */
    .progress-container {
        width: 100%;
        background-color: #1e2130;
        border-radius: 10px;
        height: 12px;
        margin: 20px 0 30px 0;
        border: 1px solid #3e445e;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff8e8e);
        transition: width 0.5s ease-in-out;
    }

# --- HEADER SECTION ---
with st.container():
    st.markdown('<div class="studio-header">', unsafe_allow_html=True)
    col_logo, col_title, col_director = st.columns([1, 3, 1])
    with col_logo:
        st.image("assets/jp.png", width=80)
        st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #a0a0a0; margin-top: -10px;'>Director</p>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<h1 style='text-align: center; margin: 0;'>🎬 Content Creating Studio</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0a0a0;'>Transform your images into professional educational reels in seconds.</p>", unsafe_allow_html=True)
    with col_director:
        st.image("assets/director.png", width=80)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- PROGRESS ROADMAP ---
    progress_percent = (st.session_state.step - 1) * 33.33
    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 10px;'>
            <div class="progress-container">
                <div class="progress-fill" style="width: {progress_percent}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
    cols_road = st.columns(4)

    # Dynamic colors based on current step
    def get_color(step_num):
        if st.session_state.step > step_num: return "🟢"
        if st.session_state.step == step_num: return "🔵"
        return "⚪"

    with cols_road[0]: st.markdown(f"{get_color(1)} **1. Import**")
    with cols_road[1]: st.markdown(f"{get_color(2)} **2. Script**")
    with cols_road[2]: st.markdown(f"{get_color(3)} **3. Voice**")
    with cols_road[3]: st.markdown(f"{get_color(4)} **4. Render**")
    st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN WORKFLOW ---
if 'step' not in st.session_state:
    st.session_state.step = 1

# Step 1: Import
if st.session_state.step == 1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">📁 Step 1: Import Source Asset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        col_img, col_action = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption="Preview Asset", width=300)
        with col_action:
            st.markdown("### ⚙️ Image Processing")

            col_extract, col_skip = st.columns(2)
            with col_extract:
                if st.button("✨ Extract Text from Image"):
                    with st.status("Analyzing Image...", expanded=True) as status:
                        image_bytes = uploaded_file.getvalue()
                        extracted = extract_text_from_image_cached(image_bytes)
                        if "OCR Error" in extracted or "API Error" in extracted:
                            st.error(extracted)
                        else:
                            st.session_state['extracted_text'] = extracted
                            status.update(label="Text Extracted!", state="complete", expanded=False)

            with col_skip:
                if st.button("Skip to Script ➡️"):
                    st.session_state['extracted_text'] = ""
                    st.session_state.step = 2
                    st.rerun()

            if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
                st.success("Text extracted successfully!")
                if st.button("Next: Refine Script ➡️"):
                    st.session_state.step = 2
                    st.rerun()
    else:
        st.info("Please upload an image to begin the workflow.")
    st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Script
elif st.session_state.step == 2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">✍️ Step 2: Refine Voice-Over Script</div>', unsafe_allow_html=True)

    if 'extracted_text' not in st.session_state:
        st.session_state['extracted_text'] = ""

    script_text = st.text_area(
        "Edit your script to make it engaging:",
        value=st.session_state['extracted_text'],
        height=300,
        label_visibility="collapsed"
    )

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back to Import"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Next: Voice Design ➡️"):
            st.session_state.script_text = script_text
            st.session_state.step = 3
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Voice
elif st.session_state.step == 3:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🎙️ Step 3: Voice & Tone Design</div>', unsafe_allow_html=True)

    col_v, col_s = st.columns(2)
    with col_v:
        voice_choice = st.selectbox("Select Voice Persona", options=list(VOICE_MAPPINGS.keys()))
    with col_s:
        speed_choice = st.selectbox("Select Speech Rate", options=list(SPEED_MAP.keys()), index=3)

    st.info("💡 **Pro Tip:** Use 'English India Female' for a natural, friendly educational tone.")

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back to Script"):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button("Next: Final Render ➡️"):
            st.session_state.voice_choice = voice_choice
            st.session_state.speed_choice = speed_choice
            st.session_state.step = 4
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Render
elif st.session_state.step == 4:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🚀 Step 4: Produce Final Video</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Final Review")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Voice:** {st.session_state.get('voice_choice')}")
        st.markdown(f"**Speed:** {st.session_state.get('speed_choice')}")
    with c2:
        st.markdown(f"**Script Preview:** {st.session_state.get('script_text', '')[:100]}...")

    if st.button("🎬 Start Production", type="primary"):
        if 'uploaded_file' not in st.session_state:
            st.error("Asset missing! Please go back to Step 1.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with st.status("Production Pipeline active...", expanded=True) as status:
                        temp_img_path = os.path.join(tmpdir, "input.png")
                        temp_audio_path = os.path.join(tmpdir, "voice.mp3")
                        temp_video_path = os.path.join(tmpdir, "output.mp4")

                        with open(temp_img_path, "wb") as f:
                            f.write(st.session_state.uploaded_file.getbuffer())

                        status.write("🎙️ Synthesizing neural voice...")
                        voice_id = VOICE_MAPPINGS[st.session_state.voice_choice]
                        rate = SPEED_MAP[st.session_state.speed_choice]
                        asyncio.run(generate_voiceover_async(st.session_state.script_text, voice_id, rate, temp_audio_path))

                        status.write("🎞️ Assembling 9:16 Reel...")
                        success = create_video_production(temp_img_path, temp_audio_path, temp_video_path)

                        if success:
                            status.update(label="Production Complete!", state="complete", expanded=False)
                            st.balloons()
                            with open(temp_video_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button("📥 Download Final MP4", data=video_bytes, file_name="content_studio_reel.mp4", mime="video/mp4")
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

    col_prev, col_reset = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back to Voice"):
            st.session_state.step = 3
            st.rerun()
    with col_reset:
        if st.button("♻️ Start Over"):
            st.session_state.step = 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Content Creating Studio v1.1 | Hybrid OCR | Neural TTS | Auto-Assemble")
