import streamlit as st
import asyncio
import edge_tts
import os
import re
import shutil
import tempfile
import pandas as pd
import requests
import io
from openai import OpenAI
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

def generate_ai_image(model_choice, topic):
    """
    Generates an AI image based on the chosen model and topic.
    Returns: (type, value) where type is 'url' or 'path'.
    """
    base_prompt = (
        f"A professional, minimalist, high-resolution educational background for {topic}. "
        "No text, no words, no letters. Clean vector art, deep dark theme, futuristic UI elements, "
        "centered empty space for text overlay, 9:16 aspect ratio, 8k, cinematic lighting, high contrast"
    )

    try:
        if model_choice == "DALL-E 3":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None, "Error: OpenAI API Key missing in .env"

            client = OpenAI(api_key=api_key)
            response = client.images.generate(
                model="dall-e-3",
                prompt=base_prompt,
                size="1024x1792",
                quality="standard",
                n=1,
            )
            return "url", response.data[0].url

        elif model_choice == "Gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None, "Error: Google API Key missing in .env"

            client = genai.Client(api_key=api_key)
            response = client.models.generate_images(
                model="imagen-3.0-generate-001",
                prompt=base_prompt,
                config={"number_of_images": 1}
            )

            # Gemini returns bytes. Save to a local file.
            img_bytes = response.generated_images[0].image._image_bytes

            # Use a stable filename in a temporary directory or assets folder
            assets_dir = "assets"
            os.makedirs(assets_dir, exist_ok=True)
            file_path = os.path.join(assets_dir, "ai_gen_background.png")

            with open(file_path, "wb") as f:
                f.write(img_bytes)

            return "path", file_path

        elif model_choice == "Pollinations (Free)":
            seed = np.random.randint(1, 1000000)
            # Pollinations uses a slightly different prompt style for flux
            poll_prompt = base_prompt.replace(" A professional", "Minimalist professional")
            url = f"https://image.pollinations.ai/prompt/{poll_prompt}?width=1080&height=1920&model=flux&seed={seed}"
            return "url", url

        return None, "Error: Unsupported model choice"
    except Exception as e:
        return None, f"Generation Error: {str(e)}"

async def generate_voiceover_async(text, voice_id, rate, output_path):
    communicate = edge_tts.Communicate(text, voice_id, rate=rate)
    await communicate.save(output_path)

def create_video_production(asset_path, audio_path, output_path):
    try:
        audio_clip = AudioFileClip(audio_path)

        # Determine if asset is a video or image
        is_video = asset_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

        if is_video:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(asset_path)
            # Loop or trim video to match audio duration
            if clip.duration < audio_clip.duration:
                # Simple loop if video is shorter than audio
                from moviepy.editor import vfx
                clip = clip.fx(vfx.loop, duration=audio_clip.duration)
            else:
                clip = clip.subclip(0, audio_clip.duration)
        else:
            clip = ImageClip(asset_path)
            if hasattr(clip, 'set_duration'):
                clip = clip.set_duration(audio_clip.duration)
            else:
                clip.duration = audio_clip.duration

        # Standardize to 9:16 (1080x1920)
        # We use "fill" logic: resize to cover the 1080x1920 area, then crop the center
        target_w, target_h = 1080, 1920

        try:
            # Calculate scale to cover target dimensions (aspect fill)
            scale = max(target_w / clip.w, target_h / clip.h)
            if hasattr(clip, 'resized'):
                clip = clip.resized(width=int(clip.w * scale), height=int(clip.h * scale))
            else:
                clip = clip.resize(width=int(clip.w * scale), height=int(clip.h * scale))
        except Exception:
            pass

        # Crop the center to exactly 1080x1920
        try:
            # Ensure we are using the latest dimensions after resize
            curr_w, curr_h = clip.size

            # Calculate the crop area to center the image
            # x1, y1 is the top-left corner of the crop
            x1 = max(0, (curr_w - target_w) // 2)
            y1 = max(0, (curr_h - target_h) // 2)

            if hasattr(clip, 'crop'):
                # MoviePy crop: x1, y1, x2, y2 (or width/height depending on version)
                # For v2.0+, it usually takes x1, y1, width, height or x1, y1, x2, y2
                # We'll use the absolute coordinates for clarity
                clip = clip.crop(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)
        except Exception:
            pass

        try:
            video = clip.set_audio(audio_clip)
        except AttributeError:
            try:
                video = CompositeVideoClip([clip])
                video = video.set_audio(audio_clip)
            except Exception:
                video = clip
                video.audio = audio_clip

        video.write_videofile(
            output_path,
            fps=12,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            preset="ultrafast",
            threads=4,
            bitrate="1000k"
        )
        audio_clip.close()
        clip.close()
        return True
    except Exception as e:
        st.error(f"Video Assembly Error: {e}")
        return False

# --- UI OVERHAUL ---
st.set_page_config(page_title="Content Creating Studio v2", layout="wide", page_icon="🎬")

st.markdown('''
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

    /* Mode Card Styling */
    .mode-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #3e445e;
        text-align: center;
        cursor: pointer;
        transition: 0.3s;
        height: 100%;
    }
    .mode-card:hover {
        border-color: #ff4b4b;
        background-color: #25293d;
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

    /* Headers */
    h1, h2, h3 { color: #ff4b4b !important; font-weight: 700 !important; }
    .step-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
    </style>
''', unsafe_allow_html=True)

# --- HEADER SECTION ---
with st.container():
    st.markdown('<div class="studio-header">', unsafe_allow_html=True)
    col_logo, col_title, col_director = st.columns([1, 3, 1])
    with col_logo:
        st.image("assets/jp.png", width=80)
        st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #a0a0a0; margin-top: -10px;'>Director</p>", unsafe_allow_html=True)
    with col_title:
        st.markdown("<h1 style='text-align: center; margin: 0;'>🎬 Content Creating Studio v2</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0a0a0;'>The next generation of AI-powered educational reels.</p>", unsafe_allow_html=True)
    with col_director:
        st.image("assets/director.png", width=80)
    st.markdown('</div>', unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'production_mode' not in st.session_state:
    st.session_state.production_mode = None

# --- NAVIGATION HELPERS ---
def switch_mode(new_mode):
    st.session_state.production_mode = new_mode
    st.session_state.step = 1 # Jump to Asset Acquisition
    st.rerun()

def reset_to_mode_selection():
    st.session_state.step = 0
    st.session_state.production_mode = None
    st.rerun()


# --- PROGRESS ROADMAP ---
# Only show roadmap if a mode is selected
if st.session_state.step > 0:
    # Mode Switcher Toolbar
    st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #1e2130; padding: 10px 20px; border-radius: 15px; border: 1px solid #3e445e;">', unsafe_allow_html=True)

    col_mode_label, col_mode_btns = st.columns([1, 3])
    with col_mode_label:
        st.markdown(f"**Current Mode:** <span style='color: #ff4b4b;'>{st.session_state.production_mode}</span>", unsafe_allow_html=True)

    with col_mode_btns:
        # Small buttons to quickly hop between modes
        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("Standard ⚡", key="nav_std", use_container_width=True):
                switch_mode("Standard")
        with btn_cols[1]:
            if st.button("Infographic 💡", key="nav_info", use_container_width=True):
                switch_mode("Infographic")
        with btn_cols[2]:
            if st.button("Cinematic 🎬", key="nav_cine", use_container_width=True):
                switch_mode("Cinematic")

    st.markdown('</div>', unsafe_allow_html=True)

    # Steps: 0: Mode, 1: Asset, 2: Script, 3: Voice, 4: Render
    # Progress is calculated from step 1 to 4
    progress_percent = (st.session_state.step - 1) * 33.33
    st.markdown(f'''
        <div style="text-align: center; margin-bottom: 10px;">
            <div class="progress-container">
                <div class="progress-fill" style="width: {progress_percent}%;"></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)


    st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
    cols_road = st.columns(4)

    def get_color(step_num):
        if st.session_state.step > step_num: return "🟢"
        if st.session_state.step == step_num: return "🔵"
        return "⚪"

    # Dynamic labels based on mode
    labels = {
        "Standard": ["Import", "Script", "Voice", "Render"],
        "Infographic": ["Concept", "Script", "Voice", "Render"],
        "Cinematic": ["Reference", "Script", "Voice", "Render"]
    }
    mode_labels = labels.get(st.session_state.production_mode, labels["Standard"])

    with cols_road[0]: st.markdown(f"{get_color(1)} **1. {mode_labels[0]}**")
    with cols_road[1]: st.markdown(f"{get_color(2)} **2. {mode_labels[1]}**")
    with cols_road[2]: st.markdown(f"{get_color(3)} **3. {mode_labels[2]}**")
    with cols_road[3]: st.markdown(f"{get_color(4)} **4. {mode_labels[3]}**")
    st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN WORKFLOW ---
# Step 0: Mode Selection
if st.session_state.step == 0:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title">🎯 Step 0: Choose Your Production Mode</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0; margin-bottom: 20px;'>Select the path that best fits your vision</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('''<div class="mode-card">
            <h3>Standard</h3>
            <p>Image &rarr; OCR &rarr; Video</p>
            <p style="font-size: 0.8rem; color: #a0a0a0;">Best for quick educational tips</p>
        </div>''', unsafe_allow_html=True)
        if st.button("Start Standard Mode"):
            st.session_state.production_mode = "Standard"
            st.session_state.step = 1
            st.rerun()

    with col2:
        st.markdown('''<div class="mode-card">
            <h3>Infographic</h3>
            <p>Title &rarr; AI Art &rarr; Video</p>
            <p style="font-size: 0.8rem; color: #a0a0a0;">Best for conceptual deep-dives</p>
        </div>''', unsafe_allow_html=True)
        if st.button("Start Infographic Mode"):
            st.session_state.production_mode = "Infographic"
            st.session_state.step = 1
            st.rerun()

    with col3:
        st.markdown('''<div class="mode-card">
            <h3>Cinematic</h3>
            <p>Ref Video &rarr; AI Video &rarr; Video</p>
            <p style="font-size: 0.8rem; color: #a0a0a0;">Best for high-end visuals</p>
        </div>''', unsafe_allow_html=True)
        if st.button("Start Cinematic Mode"):
            st.session_state.production_mode = "Cinematic"
            st.session_state.step = 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Step 1: Asset Acquisition (Branched)
elif st.session_state.step == 1:
    mode = st.session_state.production_mode
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    if mode == "Standard":
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

    elif mode == "Infographic":
        st.markdown('<div class="step-title">💡 Step 1: Define Concept</div>', unsafe_allow_html=True)
        topic = st.text_input("Enter the topic or title for your infographic:", placeholder="e.g. How Python Lists Work")

        if topic:
            st.session_state.topic = topic

            # Model Selection Feature
            model_choice = st.selectbox(
                "Select Image Generation Model",
                options=["DALL-E 3", "Gemini", "Pollinations (Free)"],
                index=0,
                help="Choose the AI provider for your background visual."
            )

            if st.button("✨ Generate AI Infographic Background"):
                with st.status("Creating AI Visual...", expanded=True) as status:
                    status.write(f"🎨 Generating visual using {model_choice}...")

                    asset_type, result = generate_ai_image(model_choice, topic)

                    if asset_type is None:
                        st.error(result)
                        status.update(label="Generation Failed", state="error")
                    else:
                        if asset_type == "url":
                            st.session_state.asset_url = result
                            st.image(result, caption=f"{model_choice} Generated Background", width=300)
                        else:
                            st.session_state.asset_path = result
                            st.image(result, caption=f"{model_choice} Generated Background", width=300)

                        status.update(label=f"{model_choice} Visual Generated!", state="complete", expanded=False)

            if ('asset_url' in st.session_state or 'asset_path' in st.session_state):
                if st.button("Next: Refine Script ➡️"):
                    # Generate a professional script based on the topic
                    st.session_state['extracted_text'] = f"Let's dive into {topic}! Understanding this concept is key to mastering the field. In this short guide, we'll break down the essentials, look at a real-world example, and see why it matters for your workflow. Let's get started!"
                    st.session_state.step = 2
                    st.rerun()

    elif mode == "Cinematic":
        st.markdown('<div class="step-title">🎬 Step 1: Upload Reference Video</div>', unsafe_allow_html=True)
        ref_video = st.file_uploader("Upload reference motion video (MP4, MOV)", type=["mp4", "mov"], label_visibility="collapsed")
        if ref_video:
            st.session_state.reference_video = ref_video
            st.video(ref_video)
            if st.button("✨ Generate Cinematic AI Video"):
                st.warning("Cinematic AI Video generation requires an API Key (Fal.ai/Replicate). Please configure .env")
                # placeholder for AI Video API call
                st.session_state.asset_path = "assets/placeholder_cinematic.mp4"
                st.success("AI Video generated (Simulation)!")

            if 'asset_path' in st.session_state:
                if st.button("Next: Refine Script ➡️"):
                    st.session_state.step = 2
                    st.rerun()
        else:
            st.info("Please upload a reference video to begin.")

    st.markdown('</div>', unsafe_allow_html=True)

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
        if 'uploaded_file' not in st.session_state and st.session_state.production_mode == "Standard":
            st.error("Asset missing! Please go back to Step 1.")
        elif st.session_state.production_mode == "Infographic" and 'asset_url' not in st.session_state:
            st.error("AI Asset missing! Please go back to Step 1.")
        elif st.session_state.production_mode == "Cinematic" and 'asset_path' not in st.session_state:
            st.error("Cinematic Asset missing! Please go back to Step 1.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with st.status("Production Pipeline active...", expanded=True) as status:
                        temp_audio_path = os.path.join(tmpdir, "voice.mp3")
                        temp_video_path = os.path.join(tmpdir, "output.mp4")
                        temp_img_path = os.path.join(tmpdir, "input.png")

                        status.write("🎙️ **Step 1/3:** Synthesizing neural voice...")
                        voice_id = VOICE_MAPPINGS[st.session_state.voice_choice]
                        rate = SPEED_MAP[st.session_state.speed_choice]
                        asyncio.run(generate_voiceover_async(st.session_state.script_text, voice_id, rate, temp_audio_path))
                        status.write("✅ Voice-over generated.")

                        status.write("🎞️ **Step 2/3:** Assembling 9:16 Reel...")
                        status.write("⏳ Encoding video frames (this may take a moment)...")

                        # Determine which asset to use based on mode
                        if st.session_state.production_mode == "Standard":
                            with open(temp_img_path, "wb") as f:
                                f.write(st.session_state.uploaded_file.getbuffer())
                            asset_path = temp_img_path
                        elif st.session_state.production_mode == "Infographic":
                            # Handle both local path (Gemini) and URL (DALL-E/Pollinations)
                            if 'asset_path' in st.session_state and os.path.exists(st.session_state.asset_path):
                                asset_path = st.session_state.asset_path
                            elif 'asset_url' in st.session_state:
                                img_data = requests.get(st.session_state.asset_url).content
                                with open(temp_img_path, "wb") as f:
                                    f.write(img_data)
                                asset_path = temp_img_path
                            else:
                                st.error("AI Asset missing! Please go back to Step 1.")
                                st.stop()
                        elif st.session_state.production_mode == "Cinematic":
                            # Use the generated AI video path
                            asset_path = st.session_state.asset_path
                        else:
                            st.error("Unknown production mode")
                            st.stop()

                        success = create_video_production(asset_path, temp_audio_path, temp_video_path)

                        if success:
                            status.write("✨ **Step 3/3:** Finalizing production...")
                            status.update(label="Production Complete!", state="complete", expanded=False)
                            st.balloons()
                            with open(temp_video_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button("📥 Download Final MP4", data=video_bytes, file_name="content_studio_reel.mp4", mime="video/mp4")
                        else:
                            status.update(label="Production Failed", state="error", expanded=True)
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

    col_prev, col_reset = st.columns([1, 1])
    with col_prev:
        if st.button("⬅️ Back to Voice"):
            st.session_state.step = 3
            st.rerun()
    with col_reset:
        if st.button("♻️ Start Over"):
            st.session_state.step = 0
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Content Creating Studio v2.0 | DALL-E 3 | Hybrid OCR | Neural TTS | Auto-Assemble")
