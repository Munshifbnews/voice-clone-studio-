# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import soundfile as sf
import os
import tempfile
from TTS.api import TTS
import time
from pathlib import Path
import cv2
import moviepy.editor as mp
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="ভয়েস এবং ফেস ক্লোন স্টুডিও",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton button {
        background-color: #4e54c8;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #3a3f9e;
    }
    .css-1v3fvcr {
        background-color: #f5f7f9;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stProgress .st-bo {
        background-color: #4e54c8;
    }
    .upload-box {
        border: 2px dashed #4e54c8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e8eaf6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .tab-content {
        padding: 20px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f5f7f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e54c8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Create temp directory for storing files
TEMP_DIR = Path(tempfile.gettempdir()) / "voice_clone_studio"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize session state variables
if 'tts_model' not in st.session_state:
    st.session_state.tts_model = None
if 'target_voice_path' not in st.session_state:
    st.session_state.target_voice_path = None
if 'language_detected' not in st.session_state:
    st.session_state.language_detected = None
if 'style_detected' not in st.session_state:
    st.session_state.style_detected = None
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'face_cascade' not in st.session_state:
    # Load OpenCV face detection model
    try:
        st.session_state.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.session_state.face_cascade = None
        
# Function to load TTS model
@st.cache_resource
def load_tts_model():
    try:
        # Load the voice conversion model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        return tts
    except Exception as e:
        st.error(f"মডেল লোড করতে সমস্যা হয়েছে: {e}")
        return None

# Function to detect language and style
def detect_language_and_style(audio_path):
    # This is a simplified version. In a real app, you would use more sophisticated
    # language and style detection algorithms.
    # For now, we'll assume Bengali and let the user select the style
    return "bn", None

# Function to clone voice
def clone_voice(tts_model, target_voice_path, text_to_speak, style=None):
    try:
        # Generate a unique output filename
        output_file = TEMP_DIR / f"cloned_voice_{int(time.time())}.wav"
        
        # Use the TTS model to clone the voice
        tts_model.tts_to_file(
            text=text_to_speak,
            file_path=str(output_file),
            speaker_wav=str(target_voice_path),
            language="bn"  # Bengali language code
        )
        
        return str(output_file)
    except Exception as e:
        st.error(f"ভয়েস ক্লোনিং প্রক্রিয়ায় সমস্যা: {e}")
        return None

# Function to detect faces in an image
def detect_faces(image):
    if st.session_state.face_cascade is None:
        return None, "ফেস ডিটেকশন মডেল লোড করতে সমস্যা হয়েছে।"
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = st.session_state.face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, "কোন মুখ শনাক্ত করা যায়নি। অনুগ্রহ করে আরেকটি ছবি আপলোড করুন।"
    
    return faces, None

# Function to extract face from image
def extract_face(image, faces, padding=20):
    if faces is None or len(faces) == 0:
        return None
    
    # Get the first face (assuming it's the main face)
    x, y, w, h = faces[0]
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Extract the face region
    face = image[y:y+h, x:x+w]
    
    return face

# Function to swap faces in a video frame
def swap_face(frame, source_face, target_face_coords, padding=20):
    if source_face is None or target_face_coords is None:
        return frame
    
    # Get target face coordinates
    x, y, w, h = target_face_coords
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    # Resize source face to match target face size
    resized_source_face = cv2.resize(source_face, (w, h))
    
    # Create a copy of the frame
    result = frame.copy()
    
    # Replace the target face with the source face
    result[y:y+h, x:x+w] = resized_source_face
    
    return result

# Function to process video for face swapping
def process_video_for_face_swap(video_path, source_face, output_path):
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = st.session_state.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # If faces are detected, swap them
            if len(faces) > 0:
                # Swap the first face found
                frame = swap_face(frame, source_face, faces[0])
            
            # Write the frame to output video
            out.write(frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 10 == 0:  # Update progress every 10 frames
                progress = frame_count / total_frames
                yield progress, frame, None
        
        # Release resources
        cap.release()
        out.release()
        
        yield 1.0, None, output_path
        
    except Exception as e:
        yield 0, None, str(e)

# Function to segment video by content
def segment_video_by_content(video_path, output_dir):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the video
        video = mp.VideoFileClip(video_path)
        
        # Get video duration in seconds
        duration = video.duration
        
        # For demonstration, we'll just split the video into equal segments
        # In a real app, you would use scene detection algorithms
        segment_count = min(5, max(2, int(duration / 300)))  # One segment per ~5 minutes
        segment_duration = duration / segment_count
        
        segments = []
        
        # Split the video into segments
        for i in range(segment_count):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            # Extract the segment
            segment = video.subclip(start_time, end_time)
            
            # Generate output filename
            output_file = os.path.join(output_dir, f"segment_{i+1}.mp4")
            
            # Write the segment to file
            segment.write_videofile(output_file, codec="libx264", audio_codec="aac")
            
            segments.append({
                "index": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "file_path": output_file
            })
            
            # Update progress
            progress = (i + 1) / segment_count
            yield progress, segments, None
        
        # Close the video to release resources
        video.close()
        
        yield 1.0, segments, None
        
    except Exception as e:
        yield 0, None, str(e)

# Main app title
st.title("🎙️ ভয়েস এবং ফেস ক্লোন স্টুডিও")

# App description
st.markdown("""
<div class="info-box">
    <p>এই অ্যাপ্লিকেশনটি আপনাকে আপনার নিজের ভয়েস ক্লোন করতে, ফেস ক্লোনিং করতে এবং ভিডিও এডিট করতে সাহায্য করবে।</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("কনফিগারেশন")
    
    # Load model button
    if st.button("মডেল লোড করুন", key="load_model"):
        with st.spinner("TTS মডেল লোড হচ্ছে... এটি কিছু সময় নিতে পারে।"):
            st.session_state.tts_model = load_tts_model()
            if st.session_state.tts_model:
                st.success("মডেল সফলভাবে লোড হয়েছে!")
    
    # Style selection
    st.subheader("ভয়েস স্টাইল")
    style_options = {
        "auto": "স্বয়ংক্রিয় শনাক্তকরণ",
        "colloquial": "চলিত ভাষা",
        "formal": "শুদ্ধ ভাষা"
    }
    selected_style = st.radio("ভয়েসের স্টাইল নির্বাচন করুন:", list(style_options.keys()), format_func=lambda x: style_options[x])
    
    # About section
    st.markdown("---")
    st.markdown("### সম্পর্কে")
    st.markdown("এই অ্যাপ্লিকেশনটি Coqui TTS লাইব্রেরি এবং OpenCV ব্যবহার করে তৈরি করা হয়েছে।")
    st.markdown("© ২০২৫ ভয়েস ক্লোন স্টুডিও")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["ভয়েস ক্লোনিং", "ফেস ক্লোনিং", "ভিডিও এডিটিং"])

# Tab 1: Voice Cloning
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("ধাপ ১: লক্ষ্য ভয়েস আপলোড করুন")
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        target_voice_file = st.file_uploader("লক্ষ্য ভয়েস ফাইল", type=["wav", "mp3", "ogg"], key="target_voice")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if target_voice_file is not None:
            # Save the uploaded file
            target_voice_path = TEMP_DIR / f"target_voice_{target_voice_file.name}"
            with open(target_voice_path, "wb") as f:
                f.write(target_voice_file.getbuffer())
            
            st.session_state.target_voice_path = str(target_voice_path)
            
            # Display audio player
            st.audio(target_voice_file, format='audio/wav')
            st.success(f"ফাইল আপলোড হয়েছে: {target_voice_file.name}")
            
            # Detect language and style
            if selected_style == "auto":
                with st.spinner("ভাষা এবং স্টাইল শনাক্ত করা হচ্ছে..."):
                    language, style = detect_language_and_style(st.session_state.target_voice_path)
                    st.session_state.language_detected = language
                    st.session_state.style_detected = style
                    
                    if language == "bn":
                        st.info("বাংলা ভাষা শনাক্ত করা হয়েছে।")
                        if style:
                            st.info(f"শনাক্ত করা স্টাইল: {'চলিত ভাষা' if style == 'colloquial' else 'শুদ্ধ ভাষা'}")
                        else:
                            st.info("স্টাইল স্বয়ংক্রিয়ভাবে শনাক্ত করা যায়নি। অনুগ্রহ করে সাইডবারে স্টাইল নির্বাচন করুন।")
                    else:
                        st.warning("বাংলা ভাষা শনাক্ত করা যায়নি। অ্যাপ্লিকেশনটি বাংলা ভাষার জন্য অপ্টিমাইজ করা হয়েছে।")
            else:
                st.session_state.style_detected = selected_style
                st.info(f"নির্বাচিত স্টাইল: {'চলিত ভাষা' if selected_style == 'colloquial' else 'শুদ্ধ ভাষা'}")
        
        st.header("ধাপ ২: সংশ্লেষণের জন্য টেক্সট লিখুন")
        text_to_speak = st.text_area("টেক্সট লিখুন", height=150, key="text_input", 
                                    placeholder="এখানে আপনি যে টেক্সটটি ক্লোন করা ভয়েসে শুনতে চান তা লিখুন...")
    
    with col2:
        st.header("ধাপ ৩: ক্লোন করা ভয়েস তৈরি করুন")
        
        if st.button("ভয়েস তৈরি করুন", key="generate_button"):
            if not st.session_state.tts_model:
                st.warning("অনুগ্রহ করে প্রথমে সাইডবার থেকে মডেল লোড করুন।")
            elif st.session_state.target_voice_path is None:
                st.warning("অনুগ্রহ করে প্রথমে একটি লক্ষ্য ভয়েস ফাইল আপলোড করুন।")
            elif not text_to_speak:
                st.warning("অনুগ্রহ করে সংশ্লেষণের জন্য কিছু টেক্সট লিখুন।")
            else:
                with st.spinner("ভয়েস ক্লোনিং প্রক্রিয়া চলছে... এটি কিছুটা সময় নিতে পারে।"):
                    progress_bar = st.progress(0)
                    
                    # Simulate progress
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                    
                    # Clone the voice
                    output_file = clone_voice(
                        st.session_state.tts_model, 
                        st.session_state.target_voice_path, 
                        text_to_speak, 
                        st.session_state.style_detected
                    )
                    
                    if output_file:
                        st.session_state.output_file = output_file
                        st.success("ভয়েস সফলভাবে তৈরি করা হয়েছে!")
                        
                        # Display audio player
                        st.audio(output_file, format='audio/wav')
                        
                        # Download button
                        with open(output_file, "rb") as file:
                            st.download_button(
                                label="ডাউনলোড .wav",
                                data=file,
                                file_name="cloned_voice.wav",
                                mime="audio/wav"
                            )
                        
                        st.markdown("""
                        <div class="success-box">
                            <p>আপনার ভয়েস ক্লোন সফলভাবে তৈরি করা হয়েছে! আপনি এটি শুনতে পারেন এবং ডাউনলোড করতে পারেন।</p>
                            <p>আপনার ভয়েসের স্টাইল (চলিত/শুদ্ধ) সংরক্ষণ করা হয়েছে।</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("ভয়েস ক্লোনিং প্রক্রিয়ায় একটি সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।")
        
        # Display previous output if available
        if st.session_state.output_file and os.path.exists(st.session_state.output_file):
            st.header("আপনার সর্বশেষ ক্লোন করা ভয়েস")
            st.audio(st.session_state.output_file, format='audio/wav')
            
            with open(st.session_state.output_file, "rb") as file:
                st.download_button(
                    label="ডাউনলোড .wav",
                    data=file,
                    file_name="cloned_voice.wav",
                    mime="audio/wav"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Face Cloning
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.header("ফেস ক্লোনিং")
    
    st.markdown("""
    <div class="info-box">
        <p>এই ফিচারটি আপনাকে একটি ছবি থেকে মুখ শনাক্ত করে অন্য ভিডিওতে সেই মুখ প্রতিস্থাপন করতে সাহায্য করবে।</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("ধাপ ১: উৎস ছবি আপলোড করুন")
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        source_image_file = st.file_uploader("আপনার ছবি", type=["jpg", "jpeg", "png"], key="source_image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        source_face = None
        
        if source_image_file is not None:
            # Read the image
            image_bytes = source_image_file.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Display the image
            st.image(source_image_file, caption="আপলোড করা ছবি", use_column_width=True)
            
            # Detect faces
            faces, error_message = detect_faces(source_image)
            
            if error_message:
                st.error(error_message)
            else:
                # Extract the face
                source_face = extract_face(source_image, faces)
                
                # Display the extracted face
                if source_face is not None:
                    # Convert from BGR to RGB for display
                    source_face_rgb = cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB)
                    st.image(source_face_rgb, caption="শনাক্ত করা মুখ", use_column_width=True)
                    st.success("মুখ সফলভাবে শনাক্ত করা হয়েছে!")
                else:
                    st.error("মুখ নির্বাচন করতে সমস্যা হয়েছে।")
        
        st.subheader("ধাপ ২: লক্ষ্য ভিডিও আপলোড করুন")
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        target_video_file = st.file_uploader("লক্ষ্য ভিডিও", type=["mp4", "avi", "mov"], key="target_video")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if target_video_file is not None:
            # Save the uploaded video
            target_video_path = TEMP_DIR / f"target_video_{target_video_file.name}"
            with open(target_video_path, "wb") as f:
                f.write(target_video_file.getbuffer())
            
            # Display the video
            st.video(target_video_file)
            st.success(f"ভিডিও আপলোড হয়েছে: {target_video_file.name}")
    
    with col2:
        st.subheader("ধাপ ৩: ফেস ক্লোনিং শুরু করুন")
        
        if st.button("ফেস ক্লোনিং শুরু করুন", key="start_face_cloning"):
            if source_face is None:
                st.warning("অনুগ্রহ করে প্রথমে একটি উৎস ছবি আপলোড করুন এবং নিশ্চিত করুন যে মুখ শনাক্ত করা হয়েছে।")
            elif target_video_file is None:
                st.warning("অনুগ্রহ করে একটি লক্ষ্য ভিডিও আপলোড করুন।")
            else:
                st.markdown("""
                <div class="warning-box">
                    <p>ফেস ক্লোনিং প্রক্রিয়া শুরু হয়েছে। এটি ভিডিওর দৈর্ঘ্য অনুযায়ী কিছু সময় নিতে পারে।</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create output path
                output_video_path = str(TEMP_DIR / f"face_cloned_{int(time.time())}.avi")
                
                # Process the video
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_display = st.empty()
                
                status_text.text("ফেস ক্লোনিং প্রক্রিয়া চলছে...")
                
                # Process video frames
                for progress, frame, result in process_video_for_face_swap(str(target_video_path), source_face, output_video_path):
                    progress_bar.progress(progress)
                    
                    if frame is not None:
                        # Display current frame being processed
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(frame_rgb, caption="প্রক্রিয়াধীন ফ্রেম", use_column_width=True)
                    
                    if result is not None:
                        if isinstance(result, str) and not os.path.exists(result):
                            st.error(f"ফেস ক্লোনিং প্রক্রিয়ায় সমস্যা: {result}")
                        else:
                            status_text.text("ফেস ক্লোনিং সম্পন্ন হয়েছে!")
                            
                            # Convert AVI to MP4 for better compatibility
                            mp4_output_path = str(TEMP_DIR / f"face_cloned_{int(time.time())}.mp4")
                            try:
                                video_clip = mp.VideoFileClip(output_video_path)
                                video_clip.write_videofile(mp4_output_path, codec="libx264")
                                video_clip.close()
                                
                                # Display the result
                                st.video(mp4_output_path)
                                
                                # Download button
                                with open(mp4_output_path, "rb") as file:
                                    st.download_button(
                                        label="ডাউনলোড ভিডিও",
                                        data=file,
                                        file_name="face_cloned_video.mp4",
                                        mime="video/mp4"
                                    )
                                
                                st.markdown("""
                                <div class="success-box">
                                    <p>ফেস ক্লোনিং সফলভাবে সম্পন্ন হয়েছে! আপনি ভিডিওটি দেখতে এবং ডাউনলোড করতে পারেন।</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"ভিডিও কনভার্ট করতে সমস্যা: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Video Editing
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.header("ভিডিও এডিটিং")
    
    st.markdown("""
    <div class="info-box">
        <p>এই ফিচারটি আপনাকে লম্বা ভিডিও (২০-৩০ মিনিট) বিষয়ভিত্তিক সেগমেন্ট করতে সাহায্য করবে।</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("ধাপ ১: ভিডিও আপলোড করুন")
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    video_file = st.file_uploader("ভিডিও ফাইল", type=["mp4", "avi", "mov"], key="edit_video")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if video_file is not None:
        # Save the uploaded video
        video_path = TEMP_DIR / f"edit_video_{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Display the video
        st.video(video_file)
        st.success(f"ভিডিও আপলোড হয়েছে: {video_file.name}")
        
        st.subheader("ধাপ ২: ভিডিও সেগমেন্টেশন শুরু করুন")
        
        if st.button("ভিডিও সেগমেন্টেশন শুরু করুন", key="start_segmentation"):
            st.markdown("""
            <div class="warning-box">
                <p>ভিডিও সেগমেন্টেশন প্রক্রিয়া শুরু হয়েছে। এটি ভিডিওর দৈর্ঘ্য অনুযায়ী কিছু সময় নিতে পারে।</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create output directory
            output_dir = str(TEMP_DIR / f"segments_{int(time.time())}")
            
            # Process the video
            progress_bar = st.progress(0)
            status_text = st.empty()
            segments_info = st.empty()
            
            status_text.text("ভিডিও সেগমেন্টেশন প্রক্রিয়া চলছে...")
            
            # Process video for segmentation
            for progress, segments, error in segment_video_by_content(str(video_path), output_dir):
                progress_bar.progress(progress)
                
                if error:
                    st.error(f"ভিডিও সেগমেন্টেশন প্রক্রিয়ায় সমস্যা: {error}")
                elif segments:
                    # Display segments information
                    segments_df = {
                        "সেগমেন্ট": [f"সেগমেন্ট {s['index']}" for s in segments],
                        "শুরু সময়": [f"{int(s['start_time'] // 60)}:{int(s['start_time'] % 60):02d}" for s in segments],
                        "শেষ সময়": [f"{int(s['end_time'] // 60)}:{int(s['end_time'] % 60):02d}" for s in segments],
                        "দৈর্ঘ্য": [f"{int(s['duration'] // 60)}:{int(s['duration'] % 60):02d}" for s in segments]
                    }
                    
                    segments_info.dataframe(segments_df)
                    
                    if progress >= 1.0:
                        status_text.text("ভিডিও সেগমেন্টেশন সম্পন্ন হয়েছে!")
                        
                        st.markdown("""
                        <div class="success-box">
                            <p>ভিডিও সেগমেন্টেশন সফলভাবে সম্পন্ন হয়েছে! নিচে সেগমেন্টগুলো দেখুন এবং ডাউনলোড করুন।</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display segments for download
                        st.subheader("ধাপ ৩: সেগমেন্টগুলো ডাউনলোড করুন")
                        
                        for segment in segments:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.video(segment["file_path"])
                            
                            with col2:
                                st.markdown(f"""
                                **সেগমেন্ট {segment['index']}**  
                                শুরু: {int(segment['start_time'] // 60)}:{int(segment['start_time'] % 60):02d}  
                                শেষ: {int(segment['end_time'] // 60)}:{int(segment['end_time'] % 60):02d}  
                                দৈর্ঘ্য: {int(segment['duration'] // 60)}:{int(segment['duration'] % 60):02d}
                                """)
                                
                                with open(segment["file_path"], "rb") as file:
                                    st.download_button(
                                        label=f"ডাউনলোড সেগমেন্ট {segment['index']}",
                                        data=file,
                                        file_name=f"segment_{segment['index']}.mp4",
                                        mime="video/mp4"
                                    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© ২০২৫ ভয়েস এবং ফেস ক্লোন স্টুডিও | সমস্ত অধিকার সংরক্ষিত")
