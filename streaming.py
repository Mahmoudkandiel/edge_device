
import streamlit as st
import cv2

# Set the page configuration
st.set_page_config(page_title="RTSP Stream Viewer", layout="centered")

st.title("Live RTSP Stream")
st.caption("Stream Source: 10.0.3.71 (Channel 401)")

# The RTSP URL provided
RTSP_URL = "rtsp://admin:Aa112233@10.0.3.71:554/Streaming/channels/401"

# Create a layout with a start/stop button
start_button = st.button("Start Stream")
stop_button = st.button("Stop Stream")

# Use session state to handle the running status
if "run" not in st.session_state:
    st.session_state["run"] = False

if start_button:
    st.session_state["run"] = True

if stop_button:
    st.session_state["run"] = False

# Placeholder for the video frame
frame_placeholder = st.empty()

# Main loop
if st.session_state["run"]:
    # Open the video capture
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        st.error("Error: Could not open video stream. Please check your network connection and credentials.")
        st.session_state["run"] = False
    else:
        while st.session_state["run"]:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("Stream ended or failed to read frame.")
                break
            
            # Convert the frame from BGR (OpenCV standard) to RGB (Streamlit standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB")
        
        # Release resources when the loop ends
        cap.release()