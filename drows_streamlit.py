import cv2
import time
import streamlit as st
import numpy as np
from PIL import Image
import os
import sys
import base64
import threading

# Add parent directory to path for Firebase import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Firebase manager
try:
    from firebase_config import firebase_manager
    firebase_manager.initialize()
    FIREBASE_ENABLED = firebase_manager.initialized
except ImportError:
    print("⚠️ Firebase config not found")
    FIREBASE_ENABLED = False
    firebase_manager = None

# Page config
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .status-safe {
        background-color: #00D084;
        color: white;
    }
    .status-alert {
        background-color: #FF4B4B;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">😴 SmartRail Shield - Pilot Drowsiness Detection</div>', unsafe_allow_html=True)

# Load Haar Cascades
@st.cache_resource
def load_cascades():
    """Load Haar Cascade classifiers from OpenCV's built-in data"""
    try:
        # Use OpenCV's built-in cascades (more reliable)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        
        # Verify they loaded correctly
        if face_cascade.empty():
            st.error("❌ Failed to load face cascade classifier")
            st.stop()
        if eye_cascade.empty():
            st.error("❌ Failed to load eye cascade classifier")
            st.stop()
            
        return face_cascade, eye_cascade
        
    except Exception as e:
        st.error(f"❌ Error loading cascade classifiers: {str(e)}")
        st.stop()

face_cascade, eye_cascade = load_cascades()

# Parameters
EYE_CLOSED_TIME = 2.0  # seconds

# Session state initialization
if 'closed_start' not in st.session_state:
    st.session_state.closed_start = None
if 'alarm_on' not in st.session_state:
    st.session_state.alarm_on = False
if 'total_alerts' not in st.session_state:
    st.session_state.total_alerts = 0
if 'play_alarm' not in st.session_state:
    st.session_state.play_alarm = False
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False

# Sidebar controls
st.sidebar.header("⚙️ Settings")
sensitivity = st.sidebar.slider("Eye Closed Time Threshold (seconds)", 0.5, 5.0, 1.5, 0.5,
    help="Time before drowsiness alert - Lower = Faster alert, Higher = More delay")
camera_index = st.sidebar.selectbox("Camera Source", [0, 1, 2], index=0)
enable_sound = st.sidebar.checkbox("🔊 Enable Alarm Sound", value=True)
eye_detection_sensitivity = st.sidebar.slider("Eye Detection Sensitivity", 1, 10, 2, 1,
    help="Lower = more sensitive (faster detection), Higher = less sensitive (fewer false alarms)")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Status")
    status_placeholder = st.empty()
    
    st.subheader("📈 Statistics")
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        alert_count = st.empty()
    with metrics_col2:
        eyes_status = st.empty()

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    start_btn = st.button("▶️ Start Detection", type="primary", use_container_width=True, disabled=st.session_state.detection_running)
with col_btn2:
    stop_btn = st.button("⏹️ Stop Detection", use_container_width=True, disabled=not st.session_state.detection_running)
with col_btn3:
    if st.button("🔄 Reset Camera", use_container_width=True):
        st.session_state.detection_running = False
        time.sleep(0.5)
        st.rerun()

if start_btn:
    st.session_state.detection_running = True
    st.rerun()
    
if stop_btn:
    st.session_state.detection_running = False
    st.rerun()

# Main detection loop
if st.session_state.detection_running:
    # Try to open camera with explicit release first
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    
    if not cap.isOpened():
        st.error("❌ Unable to open camera. Please check your camera connection or try a different camera source.")
        st.warning("💡 Tip: If another module is using the camera, stop that module first or click 'Reset Camera'")
        st.session_state.detection_running = False
    else:
        st.success("✅ Camera started successfully!")
        
        # Run detection in a loop
        frame_count = 0
        max_frames = 500  # Reduced for better responsiveness
        
        # Create a stop button placeholder that updates during loop
        stop_container = st.empty()
        
        try:
            while st.session_state.detection_running and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from camera")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                drowsy = True  # assume eyes closed
                eyes_detected = 0
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    
                    # Enhanced eye detection with multiple passes for better accuracy
                    # First pass: Strict detection
                    eyes = eye_cascade.detectMultiScale(
                        roi_gray, 
                        scaleFactor=1.1,  # Better scale stepping
                        minNeighbors=eye_detection_sensitivity, 
                        minSize=(15, 15),  # Slightly smaller for better detection
                        maxSize=(80, 80)   # Limit max size
                    )
                    
                    # If no eyes detected, try with more lenient parameters
                    if len(eyes) == 0:
                        eyes = eye_cascade.detectMultiScale(
                            roi_gray, 
                            scaleFactor=1.05, 
                            minNeighbors=max(1, eye_detection_sensitivity - 1), 
                            minSize=(12, 12)
                        )
                    
                    eyes_detected = len(eyes)
                    
                    if eyes_detected > 0:
                        drowsy = False
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(
                                roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                            )
                
                # Drowsiness Logic
                current_status = "👁️ Eyes Open - SAFE"
                status_class = "status-safe"
                
                if drowsy:
                    if st.session_state.closed_start is None:
                        st.session_state.closed_start = time.time()
                    elif time.time() - st.session_state.closed_start > sensitivity:
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3
                        )
                        current_status = "🚨 DROWSINESS ALERT!"
                        status_class = "status-alert"
                        
                        if not st.session_state.alarm_on:
                            st.session_state.alarm_on = True
                            st.session_state.total_alerts += 1
                            st.session_state.play_alarm = True
                            
                            # Save to Firebase
                            if FIREBASE_ENABLED and firebase_manager:
                                try:
                                    alert_data = {
                                        'module_name': 'Pilot Drowsiness',
                                        'alert': 'DROWSINESS_DETECTED',
                                        'eyes_detected': eyes_detected,
                                        'total_alerts': st.session_state.total_alerts
                                    }
                                    # Central PUSH
                                    firebase_manager.push_to_realtime('drowsiness_alerts', alert_data)
                                except Exception as e:
                                    print(f"⚠️ Firebase push failed: {str(e)}")
                else:
                    st.session_state.closed_start = None
                    st.session_state.alarm_on = False
                    st.session_state.play_alarm = False
                
                # Play alarm sound if enabled
                if st.session_state.play_alarm and enable_sound:
                    st.markdown("""
                    <audio autoplay>
                        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIGGS57OihUBELTKXh8LJnHgU7k9n0yXkpBSh+zPLaizsKGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsLGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsLGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsLGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsLGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsLGGe96+mjUxELTqfj8LJnHwU8lNr1yHcoBSh9y/HajDsL" type="audio/wav">
                    </audio>
                    """, unsafe_allow_html=True)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update status
                status_placeholder.markdown(
                    f'<div class="alert-box {status_class}">{current_status}</div>',
                    unsafe_allow_html=True
                )
                
                # Update metrics
                alert_count.metric("Total Alerts", st.session_state.total_alerts)
                eyes_status.metric("Eyes Detected", eyes_detected)
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.03)
                frame_count += 1
        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")
        finally:
            # Always release camera
            if cap is not None and cap.isOpened():
                cap.release()
                cv2.destroyAllWindows()  # Cleanup any OpenCV windows
            st.session_state.detection_running = False
            st.info("⏹️ Detection stopped - Camera released")
            
            # If max frames reached, restart
            if frame_count >= max_frames:
                time.sleep(0.5)
                st.rerun()

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Instructions:
    1. **Click "Start Detection"** to begin monitoring
    2. The system will detect your face and eyes in real-time
    3. If eyes are closed for more than the threshold time, an alert will be triggered
    4. **Adjust sensitivity** in the sidebar to change the alert threshold
    5. **Click "Stop Detection"** to end the session
    
    ### Features:
    - ✅ Real-time face and eye detection
    - ✅ Customizable drowsiness threshold
    - ✅ Alert counter and statistics
    - ✅ Visual alerts on screen
    - ✅ Audio alarm support
    
    ### Safety Tips:
    - Ensure good lighting for better detection
    - Position camera at eye level
    - Avoid wearing sunglasses or hats that cover eyes
    
    ### Troubleshooting:
    - If camera doesn't open, try changing camera source in sidebar
    - Adjust eye detection sensitivity if false alerts occur
    - Ensure proper lighting for best results
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>SmartRail Shield - Drowsiness Detection Module v2.0</div>",
    unsafe_allow_html=True
)
